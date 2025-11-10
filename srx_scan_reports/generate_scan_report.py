import numpy as np
import time as ttime
import os
import json

from PyPDF2 import PdfReader, PdfWriter, PdfMerger

from tiled.queries import Key
from httpx import ReadTimeout

from .SRXScanPDF import SRXScanPDF


def generate_scan_report(start_id=None,
                         end_id=None,
                         proposal_id=None,
                         cycle=None,
                         wd=None,
                         continuous=True,
                         verbose=False,
                         **kwargs):
    
    """
    Function for generating an SRX scan report.

    Function for generating an SRX scan report based on information
    provided by the start_id, end_id, proposal_id, and cycle and the
    continuous flag. Several combinations exists dictating different
    behavior depending on which parameters are provided:

    - If the start_id is provided and end_id is not, then every scan
      will be added to the report within the proposal scope of the
      start_id. If the continuous flag is False, the report will stop
      after the first change in the proposal scope.
    - If both the start_id and end_id are provided, then every scan
      between the two will be appended to the report, if the scan
      matchesthe proposal scope of the start_id.
    - If the start_id is not provided and proposal_id and cycle are,
      then the first scan ID matching the proposal scope will be used
      as the start_id. If the most recent scan ID falls outside the
      proposal scope and scans already exist within the proposal scope,
      then the last scan within the proposal scope will be set as the
      end_id, and the continuous flag will be set to True. The
      combinations of these two definitions are defined as above.
    - The continuous flag dictates how breaks in the proposal scope are
      handled. If False, then report generation will be paused when a
      scan ID is encountered that leaves the proposal scope. This
      really only affects the first bullet in this list.
    - If neither the start_id or proposal_id and cycle are provided, a
      ValueError will be raised based on insufficient information to
      begin the scan report.

    Scan report generation will initiate a loop that will continuously
    monitor scan completion and append finished scans. At any time
    during this loop, a KeyboardInterrupt will pause the report
    generation. Waiting 10 seconds after the KeyboardInterrupt will
    leave the report in a paused state, which can be resumed by
    running the function again with the same start_id, end_id,
    proposal_id, and cycle parameters. If another KeyboardInterrupt
    is entered within this 10 sec window, the report will be finalized
    and new scans can no longer be appended.


    Parameters
    ----------
    start_id : int, optional
        Scan ID of first scan included in the scan report. If this scan
        ID exists in the database, the proposal ID and cycle
        information will be taken for this scan ID. If this parameter
        is None, then the first scan ID matching the proposal ID and
        cycle will be used instead.
    end_id : int, optional
        Scan ID of the last scan included in the scan report. If this
        parameter is provided, report generation will continue from the
        start ID until the end ID.
    proposal_id : int, optional
        The ID of the proposal used with the cycle to determine the
        proposal scope. If None, the proposal_id will be set base on
        the start_id scan. 
    cycle : str, optional
        The cycle of the proposal used with the proposal_id to
        determine the proposal scope. If None, the cycle will be set
        based on the start_id scan.
    continuous : bool, optional
        This flag dicates how breaks in the proposal scope are handled.
        If False, then report generation will be paused when a scan ID
        leaves the proposal scope. If True, scan IDs outside the
        proposal scope will be skipped and scan IDs that re-enter the
        proposal scope will be appended. This only affects behavior
        when the start_id is specified and the end_id is None.
    wd : path, optional
        Path to write the scan_report. If None, a path will be built
        from the proposal_id and cycle if provided, or from the
        same information determined from the start_id. If 'scan_report'
        is not within the wd or one of its sub-directories, a new
        'scan_reports' will be created within wd.
    verbose : bool, optional
        Flag to determine the verbosity. False by default.
    kwargs : dict, optional
        Keyword argmuents passed to the 'add_scan' function and other
        functions that are called within.

        add_scan
        --------
            include_optimizers : bool, optional
                Flag to include PEAKUP and OPTIMIZE_SCALERS functions
                in the scan report. True by default.
            include_unknowns : bool, optional
                Flag to include unknown scan types in the scan report.
                True by default.
            include_failures : bool, optional
                Flag to include failed and aborted scans in the scan
                report. True by default.

        add_XRF_FLY
        -----------
            min_roi_num : int, optional
                Minimum number of rois to try and include for fly scan
                plots. Should not go below 1 or exceed max_roi_num.
                This parameter cannot force plotting of elements that
                do not exist and may not currently do anything.
            max_roi_num : int, optional
                Maximmum number of rois to try and include for fly scan
                plots. Should not exceed 10 and works well for 1, 4, 7,
                and 10. By default this value is 10.
            scaler_rois : list, optional
                List of scaler names to include in the fly scan plots.
                Accepts 'i0', 'im', and 'it', by will include none by
                default.
            ignore_det_rois : list, optional
                List of detectors names as strings to be ignored for
                fly scan plotting. For example ['dexela'], but will be
                empty by default.
            colornorm : str, optional
                Color normalization passed to matplotlib.axes.imshow.
                Can be 'linear', or 'log'. 'linear' by default.

        _find_xrf_rois
        --------------
            specific_elements : list, optional
                List of element abbreviations as strings which will be
                guaranteed inclusion in the report. Theses elements
                will appear in the order given. By default, no elements
                will be guaraneteed.
            log_prominence : float, optional
                The prominence value passed to scipy.signal.find_peaks
                function of the log of the integrated XRF spectra.
            energy_tolerance : float, optional
                Energy window used for peak assignment of identified
                peaks. 50 eV by default.
            esc_en : float, optional
                Escape energy in eV. Used to allow for escape peak
                assignment. 1740 (for Si detectors) by default.
            snr_cutoff : float, optional
                Signal-to-noise ratio cutoff value. Use to determine
                the cutoff value for thresholding peak significance.


    Raises
    ------
    ValueError if insufficient information is provided to determine the
        report write location, or if the start_id is greater than the
        end_id.
    TypeError if key in kwargs is not expected.
    
    Examples
    --------
    After loading this file.

    >>> generate_scan_report(12345, continuous=False, max_roi_num=4,
    ... specific_elements=['Fe', 'Cr', 'Ni'])

    This will start generating a report starting with scan ID 12345 and
    will continue until the scans leave the proposal scope of this scan
    ID or until a KeyboardInterrupt is given. This scan report will
    plot up to 4 regions of interest for each XRF_FLY scan and will try
    to include 'Fe', 'Cr', and 'Ni' first in that order. If no area
    detectors are included, the report will try to include another
    element to fill the 4 regions of interest, but this may not succeed
    depending on the XRF signal.
    """

    # Quick function to get keyword argument names
    get_kwargs = lambda func : func.__code__.co_varnames[func.__code__.co_argcount - len(func.__defaults__) : func.__code__.co_argcount]

    # Parse kwargs to make sure they are useful. Add functions and methods as necessary
    useful_kwargs = (list(get_kwargs(SRXScanPDF.add_scan))
                     + list(get_kwargs(SRXScanPDF._add_xrf_general))
                     + list(get_kwargs(_find_xrf_rois)))
    useful_kwargs.remove('scan_kwargs')
    for key in kwargs.keys():
        if key not in useful_kwargs:
            err_str = f"generate_scan_report got an unexpected keyword argument '{key}'"
            raise TypeError(err_str)
    
    # Parse requested inputs
    # Get data from start id first
    if start_id is not None:
        if start_id == -1:
            start_id = int(c[-1].start['scan_id'])
        else:
            start_id = int(start_id)

        if start_id in c:
            start_cycle = c[start_id].start['cycle']
            if 'proposal_id' in c[start_id].start['proposal']:
                start_proposal_id = c[start_id].start['proposal']['proposal_id']
            elif 'proposal_num' in c[start_id].start['proposal']:
                start_proposal_id = c[start_id].start['proposal']['proposal_num']
            else:
                err_str = f'Cannot find proposal identification key from start document!'
                raise ValueError(err_str)

            if ((proposal_id is not None and proposal_id != start_proposal_id)
                or (cycle is not None and cycle != start_cycle)):
                warn_str = (f'WARNING: Starting scan ID of {start_id} '
                            + f'has proposal ID ({start_proposal_id}) '
                            + f'and cycle ({start_cycle}) which does '
                            + f'not match the provided proposal ID '
                            + f'({proposal_ID}) and cycle ({cycle})!'
                            + f'\nUsing the start ID {start_id} '
                            + 'information.')
                print(warn_str)
                proposal_id = start_proposal_id
                cycle = start_cycle
            else:
                proposal_id = start_proposal_id
                cycle = start_cycle
            
            if wd is None:
                wd = f'/nsls2/data3/srx/proposals/{cycle}/pass-{proposal_id}'
        
        else:
            if wd is None:
                err_str = ('Cannot determine write location. Please '
                           + 'provide start_id of previous scan or '
                           + 'cycle and proposal_id.')
                raise ValueError(err_str)               

    # Default to proposal information next. This may be more popular
    elif proposal_id is not None and cycle is not None:
        if wd is None:
            wd = f'/nsls2/data3/srx/proposals/{cycle}/pass-{proposal_id}'
        lim_c = c.search(Key('cycle') == str(cycle)).search(Key('proposal.proposal_id') == str(proposal_id))
        if len(lim_c) > 0:
            start_id = int(lim_c[0].start['scan_id'])
            if end_id is None: # Attempt to see if the proposal is already finished
                last_id = int(c[-1].start['scan_id'])
                end_id = int(lim_c[-1].start['scan_id'])
                if last_id == end_id:
                    end_id = None
                continuous = True
        else:
            # Hoping the next scan will be correct
            start_id = int(c[-1].start['scan_id']) + 1
    
    else:
        err_str = ('Cannot determine write location. Please provide'
                   + ' start_id of previous scan or cycle and '
                   + 'proposal_id.')
        raise ValueError(err_str)
    
    # Pre-populate proposal information
    exp_md = {'proposal_id' : proposal_id,
              'cycle' : cycle}
  
    if end_id is not None and end_id < 0:
        end_id = int(c[int(end_id)].start['scan_id'])
        continuous = True
    elif end_id is not None:
        end_id = int(end_id)
        if end_id < start_id:
            err_str = (f'end_id ({end_id}) of must be greater than or '
                       + f'equal to start_id ({start_id}).')
            raise ValueError(err_str)
    current_id = start_id # Create current_id as counter

    if verbose:
        print(f'Final end_id is {end_id}')
    
    # Setup file paths
    if 'scan_report' not in wd: # Lacking the 's' for generizability
        directories = [x.path for x in os.scandir(wd) if x.is_dir()]
        for d in directories:
            if 'scan_report' in d:
                wd = os.path.join(wd, d)
                break
        else:
            wd = os.path.join(wd, 'scan_reports')
    os.makedirs(wd, exist_ok=True)

    # Setup filename and exact paths
    filename = f'scan{start_id}-{end_id}_report'
    pdf_path = os.path.join(wd, f'{filename}.pdf')
    md_path = os.path.join(wd, f'{filename}_temp_md.json')

    # Read pdf and md if exists
    current_pdf = None
    pdf_md = None
    # Only append to reports if the temp_md.json file also exists
    if os.path.exists(pdf_path) and os.path.exists(md_path):
        current_pdf = PdfReader(pdf_path)
        with open(md_path) as f:
            pdf_md = json.load(f)
        current_id = pdf_md['current_id'] + 1 # This off-by one may be wrong...
    
    if verbose:
        print(f'PDF path is: {pdf_path}')
    
    # Move to continuous writing
    wait_iter = 0
    while True:
        try:
            # First check if the scan report has finished
            if end_id is not None and current_id > end_id:
                os.remove(md_path)
                print(f'Finishing scan report...')
                break
            
            # Second check to see if the current_id is finished
            WAIT = False
            recent_id = c[-1].start['scan_id']
            # current_id has not been started
            if current_id > recent_id: 
                WAIT = True
            elif current_id <= recent_id:
                if current_id not in c:
                    err_str = (f'Scan ID {current_id} not found in '
                               + 'tiled. Currently no provisions for '
                               + 'this error.\nSkipping and moving to '
                               + 'next scan ID.')
                    current_id += 1
                    continue
                # Has the current_id finished?
                elif current_id == recent_id:
                    if (not hasattr(c[current_id], 'stop')
                        or c[current_id].stop is None
                        or 'time' not in c[current_id].stop):
                        WAIT = True
            
            # Current scan has yet to finish. Give it some time.
            if WAIT:
                wait_time = 60
                print(f'Current scan ID {current_id} not yet finished. {wait_iter} minutes since last scan added...', end='\r', flush=True)
                ttime.sleep(wait_time)
                wait_iter += 1
                continue
            else:
                if wait_iter > 0:
                    print('', flush=True)
                    wait_iter = 0
            
            # Third check if the current_id is within the proposal
            scan_report = SRXScanPDF(verbose=verbose)
            scan_report.get_proposal_scan_data(current_id)
            if all([scan_report.exp_md[key] == exp_md[key]
                    for key in ['proposal_id', 'cycle']]):
                exp_md = scan_report.exp_md
            else:
                note_str = (f'Current scan ID {current_id} not within '
                           + f'Proposal # : {exp_md["proposal_id"]}.')
                print(note_str)
                # Continuously appending or waiting to initialize
                if continuous or current_pdf is None:
                    print(f'\tSkipping Scan {current_id}.')
                    current_id += 1
                    continue
                # Already initialized, expectation is to be finished.
                else:
                    os.remove(md_path)
                    print(f'Finishing scan report...')
                    break                

            # Final check performed on first successful current_id only
            if current_pdf is None:
                print(f'Initializing scan report...')
                # Create first pdf page
                scan_report.add_page()

                # Update md
                pdf_md = {'current_id' : current_id,
                          'abscissa' : (scan_report.x, scan_report.y)}

                # Write data
                scan_report.output(pdf_path)
                with open(md_path, 'w') as f:
                    json.dump(pdf_md, f)

                # Read data. # md already loaded
                current_pdf = PdfReader(pdf_path)

                # Regenerate scan report
                scan_report = SRXScanPDF(verbose=verbose)
                scan_report.exp_md = exp_md
            
            print(f'Adding scan {current_id}...')
            scan_report._appended_pages = current_pdf.numPages - 1

            # Add blank page for overlay
            scan_report.add_page(disable_header=True)

            # Set cursor location
            scan_report.set_xy(*pdf_md['abscissa'])
            # Add new scan
            try:
                scan_report.add_scan(current_id, **kwargs)
            except ReadTimeout as e:
                print(f'Error encountered reading data.\n\tReadTimeout: {e}')
                print('Waiting for 1 minute and trying again...')
                ttime.sleep(60)
                continue
            # Update md
            pdf_md = {'current_id' : current_id,
                      'abscissa' : (scan_report.x, scan_report.y)}

            # Overlay first page of new pdf to last page of previous
            new_pdf = PdfReader(io.BytesIO(scan_report.output()))
            current_pdf.pages[-1].merge_page(page2=new_pdf.pages[0])
            
            # Add these pages to the new writer
            writer = PdfWriter()
            writer.append_pages_from_reader(current_pdf)
            # Add any newly generated pages
            for i in range(1, new_pdf.numPages):
                writer.add_page(new_pdf.pages[i])
            # Overwrite pervious file with updated pdf
            writer.write(pdf_path)

            # Overwrite previous data
            with open(md_path, 'w') as f:
                json.dump(pdf_md, f)

            # Re-read data to update current_pdf
            # md is already updated
            current_pdf = PdfReader(pdf_path)

            # Update scan_id
            current_id += 1

        except KeyboardInterrupt:
            try:
                print('') # for the '^C'
                print('KeyboardIterrupt triggered; report generation paused. Waiting 10 sec before exiting...')
                print('Press ctrl+C again to finalize and cleanup report in its current state.')
                ttime.sleep(10)
                break
            except KeyboardInterrupt:
                # Cleanup files
                print('') # for the '^C'
                print(f'Report generation finalized on scan {current_id}. Cleaning up files in their current state...')
                new_filename = f'scan{start_id}-{current_id - 1}_report'
                new_pdf_path = os.path.join(wd, f'{new_filename}.pdf')
                os.rename(pdf_path, new_pdf_path)
                os.remove(md_path)
                break
        except Exception as e:
            print(f'Error encountered for scan {current_id}. Pausing report generation.')
            raise e

    print('done!')
