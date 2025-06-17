import numpy as np
import time as ttime
from tqdm import tqdm

from fpdf import FPDF
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from PIL import Image
from scipy.signal import find_peaks
import skbeam.core.constants.xrf as xrfC
from itertools import combinations_with_replacement
from scipy.ndimage import median_filter
from scipy.stats import mode


from tiled.client import from_profile

# Access data
c = from_profile('srx')

# Get elemental information
possible_elements = ['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                     'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                     'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                     'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                     'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                     'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                     'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                     'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

boring_elements = ['Ar']

elements = [xrfC.XrfElement(el) for el in possible_elements]
edges = ['k', 'l1', 'l2', 'l3']
lines = ['ka1', 'kb1', 'la1', 'lb1', 'lb2', 'lg1', 'la2', 'lb3', 'lb4', 'll', 'ma1', 'mb']
major_lines = ['ka1', 'la1', 'lb1', 'ma1']
roi_lines = ['ka1', 'la1', 'ma1']

all_edges, all_edges_names, all_lines, all_lines_names = [], [], [], []
for el in elements:
    for edge in edges:
        edge_en = el.bind_energy[edge] * 1e3
        if 4e3 < edge_en < 22e3:
            all_edges.append(edge_en)
            all_edges_names.append(f'{el.sym}_{edge.capitalize()}')

# No provisions to handle peak overlaps
# Also only gives rois for major XRF lines (ka1, la1, ma1)
def _find_xrf_rois(xrf,
                   energy,
                   incident_energy,
                   specific_elements=None,
                   min_roi_num=0,
                   max_roi_num=25, # Hard capping for too many elements
                   log_prominence=1,
                   energy_tolerance=50,
                   esc_en=1740,
                   snr_cutoff=100,
                   scan_kwargs={}):

        # Parse out scan_kwargs
        if 'specific_elements' in scan_kwargs:
            specific_elements = scan_kwargs,pop('specific_elements')
        if 'min_roi_num' in scan_kwargs:
            min_roi_num = scan_kwargs.pop('min_roi_num')
        if 'max_roi_num' in scan_kwargs:
            min_roi_num = scan_kwargs.pop('max_roi_num')
        if 'log_prominence' in scan_kwargs:
            log_prominence = scan_kwargs.pop('log_prominence')
        if 'energy_tolerance' in scan_kwargs:
            energy_tolerance = scan_kwargs.pop('energy_tolerance')
        if 'esc_en' in scan_kwargs:
            esc_en = scan_kwargs.pop('esc_en')
        if 'snr_cutoff' in scan_kwargs:
            snr_cutoff = scan_kwargs['snr_cutoff']

        # Parse some inputs
        if min_roi_num > max_roi_num:
            max_roi_num = min_roi_num
        
        # Do not bother if no rois requested
        if max_roi_num == 0:
            return [], [], [], []

        # Process specified elements before anything else
        found_elements = []
        num_interesting_rois = 0
        if specific_elements is not None:
            for el in specific_elements:
                if (isinstance(el, str)
                    and el.capitalize in possible_elements
                    and el not in found_elements):
                    found_elements.append(xrfC.XrfElement(el))
                    num_interesting_rois += 1

        # Do not modify the original data
        xrf = xrf.copy()

        # Convert energy to eV
        if energy[int((len(energy) - 1) / 2)] < 1e3:
            energy = np.array(energy) * 1e3
        if incident_energy < 1e3:
            incident_energy *= 1e3

        # Get energy step size
        en_step = np.mean(np.diff(energy), dtype=int)

        # Crop XRF and energy to reasonable limits (Assuming 10 eV steps)
        # No peaks below 1000 eV or above 85% of incident energy
        en_range = slice(int(1e3 / en_step), int(0.85 * incident_energy / en_step))
        xrf = xrf[en_range]
        energy = energy[en_range]

        # Remove invalid data_points (mostly zeros)
        xrf[xrf < 1] = 1

        # Estimate background and noise
        bkg = mode(xrf)[0] * (200 / en_step)
        noise = np.sqrt(bkg)

        # Find peaks based on log prominence
        peaks, proms = find_peaks(np.log(xrf), prominence=log_prominence)

        # Blindly find intensity from 200 eV window (assuming 10 eV steps)
        peak_snr = [(np.sum(xrf[peak - int(100 / en_step) : peak + int(100 / en_step)]) - bkg) / noise for peak in peaks]

        # Sort the peaks by snr
        sorted_peaks = [x for y, x in sorted(zip(peak_snr, peaks), key=lambda pair: pair[0], reverse=True)]
        sorted_snr = sorted(peak_snr, reverse=True)

        # Convert peaks to energies
        peak_energies = [energy[peak] for peak in sorted_peaks]


        # Identify peaks
        peak_labels = []
        for peak_ind, peak_en in enumerate(peak_energies):
            
            # Conditions to stop processing
            if (num_interesting_rois == max_roi_num # reached max roi count
                or (sorted_snr[peak_ind] < snr_cutoff # peak snr is now below cutoff
                    and num_interesting_rois >= min_roi_num)): # and enough rois have been identified
                break

            PEAK_FOUND = False
            # First, check if peak can be explained by an already identified element
            for el in found_elements:
                if isinstance(el, int): # Unknown peak...
                    continue

                for line in lines:
                    line_en = el.emission_line[line] * 1e3
                    # Direct peak
                    if np.abs(peak_en - line_en) < energy_tolerance:
                        PEAK_FOUND = True
                        peak_labels.append(f'{el.sym}_{line}')
                        break
                else:
                    continue
                break
            
            # Second, check if peak is artifact of already identified peaks
            # Check escape peaks first
            if not PEAK_FOUND:
                for found_peak_en, found_peak_label in zip(peak_energies[:peak_ind], peak_labels):
                    # Ignore escape peaks of other artifacts
                    if found_peak_label.split('_')[-1] in ['escape', 'sum', 'double']:
                        continue

                    if np.abs(peak_en - (found_peak_en - esc_en)) < energy_tolerance * 2: # double for error propagation
                        PEAK_FOUND = True
                        peak_labels.append(f'{found_peak_label}_escape')
                        break
            
            # Now check if it is a pile-up or sum peak
            if not PEAK_FOUND:
                for comb_ind in combinations_with_replacement(range(peak_ind), r=2): # not considering combinations above two peaks
                    sum_en = sum([peak_energies[ind] for ind in comb_ind])

                    # Kick out any combinations of other artifacts
                    if any([peak_labels[ind].split('_')[-1] in ['escape', 'sum', 'double'] for ind in comb_ind]):
                        continue
                    
                    if comb_ind[0] != comb_ind[1]:
                        comb_label = f'{peak_labels[comb_ind[0]]}_{peak_labels[comb_ind[1]]}_sum'
                    else:
                        comb_label = f'{peak_labels[comb_ind[0]]}_double'
                    
                    if np.abs(peak_en - sum_en) < energy_tolerance * 2: # double for error propagation
                        PEAK_FOUND = True
                        peak_labels.append(comb_label)
                        break

            # Otherwise check other elements based on strongest fluorescence
            if not PEAK_FOUND:
                for line in major_lines: # Major lines should always be present if elements exists
                    for el in elements:
                        line_en = el.emission_line[line] * 1e3
                        if np.abs(peak_en - line_en) < energy_tolerance:
                            PEAK_FOUND = True
                            found_elements.append(el)
                            peak_labels.append(f'{el.sym}_{line}')
                            if el.sym not in boring_elements:
                                num_interesting_rois += 1
                            break
                    else:
                        continue
                    break
                else: # Very unlikely
                    found_elements.append(int(peak_en))
                    peak_labels.append('Unknown')
                    num_interesting_rois += 1

        # Generate new ROIS
        rois, rois_labels = [], []
        for el in found_elements:
            if isinstance(el, int):
                rois.append(slice(int((el / en_step) - (100 / en_step)), int((el / en_step) + (100 / en_step))))
                rois_labels.append('Unknown')
                continue
            
            # Ignore argon mostly
            elif el.sym in boring_elements:
                continue

            # Slice major lines
            for line in roi_lines:
                line_en = el.emission_line[line] * 1e3
                if 1e3 < line_en < incident_energy:
                    rois.append(slice(int((line_en / en_step) - (100 / en_step)), int((line_en / en_step) + (100 / en_step))))
                    rois_labels.append(f'{el.sym}_{line}')
                    break

        print(peak_energies)
        print(peak_labels)

        return rois, rois_labels, peak_energies, peak_labels, # sorted_snr


class SRXScanPDF(FPDF):

    def __init__(self):
         FPDF.__init__(self)
         self.exp_md = {}
         self.set_auto_page_break(False)

    def header(self):
        self.set_font('helvetica', size=11)

        self.image('srx_logo.png', h=15, keep_aspect_ratio=True)
        self.set_xy(self.l_margin, self.t_margin)

        with self.table(
                line_height=5,
                borders_layout='NONE',
                align='R',
                width=self.epw - 15,
                text_align=('RIGHT', 'RIGHT', 'LEFT')
            ) as table:
            table.row().cell(f"Proposal # : {self.exp_md['proposal_id']}")
            table.row().cell(f"Proposal PI : {self.exp_md['pi_name']}")

            # Shrink title as necessary
            title_words = self.exp_md['title'].split(' ')
            title_text = title_words[0]
            for word in title_words[1:]:
                if len(title_text) + len(word) + 4 < 85:
                    title_text += f' {word}'
                else:
                    title_text += '...'
                    break
            table.row().cell(f"Proposal Title : {title_text}")

        self.set_line_width(0.5)
        self.set_draw_color(0, 0, 0)
        self.line(x1=self.l_margin, y1=self.y + 1.5, x2=self.epw + self.l_margin, y2=self.y + 1.5)

    def footer(self):
        self.set_xy(self.x, -15)
        self.set_font('helvetica', size=10)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        self.cell(0, 10, f'Generated on {ttime.ctime()}', align='R')        

    
    def add_scan(self,
                 scan_id,
                 include_peakups=True,
                 include_unknowns=True,
                 include_failures=True,
                 **kwargs):
        """Add a scan from a scan_id"""

        print(f'Adding scan {scan_id}')

        scan_kwargs = kwargs
              
        # Add first page if empty
        if self.exp_md == {}:
            self.get_proposal_scan_data(scan_id)
            self.add_page()
        else:
            # Check and update metadata anyway
            # TODO: Is this still needed?
            self.get_proposal_scan_data(scan_id)

        bs_run = c[int(scan_id)]
        # Extract metadata from start
        scan_data = self._get_start_scan_data(bs_run)
        # Extract reference data from baseline
        scan_data.update(self._get_baseline_scan_data(bs_run))
        
        # Skip failed scans if not including failures
        if (not include_failures
            and scan_data['exit_status'] != 'success'):
            return

        # Build report entry
        if scan_data['scan_type'] == 'XRF_FLY':
            start_y = self.y
            self.add_XRF_FLY(bs_run, scan_data, scan_kwargs)
            end_y = self.y
            print(f'Cell took {end_y - start_y} mm.')
        elif scan_data['scan_type'] == 'XAS_STEP':
            self.add_XAS_STEP(bs_run, scan_data, scan_kwargs)
        # elif scan_data['scan_type'] == 'XRF_STEP':
        #     self.add_XRF_STEP(bs_run, scan_data, scan_kwargs)
        # elif scan_data['scan_type'] == 'ANGLE_RC':
        #     self.add_ANGLE_RC(bs_run, scan_data, scan_kwargs)
        # elif scan_data['scan_type'] == 'ENERGY_RC':
        #     self.add_ENERGY_RC(bs_run, scan_data, scan_kwargs)
        # elif scan_data['scan_type'] == 'XAS_FLY':
        #     self.add_XAS_FLY(bs_run, scan_data, scankwargs)
        elif scan_data['scan_type'] == 'PEAKUP':
            if include_peakups:
                if self.h - self.y - self.b_margin < 15:
                    self.add_page()
                self.add_BASE_SCAN(scan_data, scan_kwargs)
            else:
                return
        elif include_unknowns:
            warn_str = (f"WARNING: Scan {scan_id} of type {scan_data['scan_type']} "
                        + "not yet support for SRX scan reports.")
            print(warn_str)
            if self.h - self.y - self.b_margin < 15:
                self.add_page()
            self.add_BASE_SCAN(scan_data, scan_kwargs)
        else:
            return
    

    def add_scan_range(self,
                       start_id,
                       end_id=-1,
                       **kwargs):
        """Add scans across a specified range"""

        end_id = c[end_id].start['scan_id']

        if end_id <= start_id:
            err_str = f'End ID of {end_id} must be larger than Start ID of {start_id}.'
            raise ValueError(err_str)

        # Get header metadata if it is empty
        if self.exp_md == {}:
            self.get_proposal_scan_data(start_id)
            self.add_page() # start first page

        for scan_id in tqdm(range(int(start_id), int(end_id) + 1)):
            self.add_scan(scan_id, **kwargs)

    
    def add_BASE_SCAN(self,
                      scan_data,
                      scan_kwargs):
        """Add basic data for all scans"""

        # Generate starting line
        self.set_line_width(0.5)
        self.set_draw_color(0, 0, 0)
        self.set_xy(self.x, self.y + 1.5)
        self.line(x1=self.l_margin, y1=self.y, x2=self.epw + self.l_margin, y2=self.y)

        scan_labels = ['ID', 'Type', 'Status', 'Start', 'Stop', 'Duration']
        scan_values = [str(scan_data['scan_id']),
                       scan_data['scan_type'],
                       scan_data['exit_status'].upper(),
                       ttime.strftime('%b %d %H:%M:%S', ttime.localtime(scan_data['start_time'])),
                       ttime.strftime('%b %d %H:%M:%S', ttime.localtime(scan_data['stop_time'])),
                       ttime.strftime('%H:%M:%S', ttime.gmtime(scan_data['stop_time'] - scan_data['start_time']))]
        
        # General scan data
        self.set_xy(self.x, self.y + 1.5)
        self.set_line_width(0.1)
        self.set_font(size=10)
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=5,
                align='L',
                ) as table:
            row = table.row()
            for i in range(len(scan_labels)):
                if i == 0:
                    self.set_font(style='B', size=10)
                row.cell(f'{scan_labels[i]}')
                if i == 0:
                    self.set_font(style='', size=10)
            row = table.row()
            for i in range(len(scan_labels)):
                if i == 0:
                    self.set_font(style='B', size=10)
                row.cell(str(scan_values[i]))
                if i == 0:
                    self.set_font(size=10)
        
        # Do not add reference positions for failed scans or peakups?
        if (scan_data['scan_type'] in ['PEAKUP', 'OPTIMIZE_SCALERS']):
            return

        ref_labels = ['Energy', 'Coarse X', 'Coarse Y', 'Coarse Z', 'Top X', 'Top Z', 'Theta', 'Scanner X', 'Scanner Y', 'Scanner Z', 'SDD Offset']
        ref_values = [scan_data[key] for key in ['energy', 'x', 'y', 'z', 'topx', 'topz', 'th', 'sx', 'sy', 'sz', 'sdd_x']]
        ref_units = [scan_data[f'{key}_units'] for key in ['energy', 'x', 'y', 'z', 'topx', 'topz', 'th', 'sx', 'sy', 'sz', 'sdd_x']]
        ref_precision = [0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0]
        
        # Reference data
        self.set_xy(self.x, self.y + 1)
        reset_y = self.y
        self.set_font(style='B', size=10)
        self.cell(h=5, new_x='LMARGIN', new_y='NEXT', text=f"Reference Data")
        self.set_font(size=9)
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=4.5,
                col_widths=(20, 20),
                width=40,
                align='L'
                ) as table:
            for i in range(len(ref_labels)):
                row = table.row()
                row.cell(ref_labels[i])
                if ref_values[i] is not None:
                    row.cell(f'{np.round(ref_values[i], ref_precision[i]):.{ref_precision[i]}f}' + f' {ref_units[i]}')
                else:
                    row.cell('-')
            table_width = table._width
        self.set_xy(self.x + table_width, reset_y)

    
    def add_XRF_FLY(self,
                    bs_run,
                    scan_data,
                    scan_kwargs,
                    min_roi_num=1,
                    max_roi_num=10, # Hard capping for too many elements
                    ignore_det_rois=[]
                    ):
        """Add data specific to XRF_FLY scans"""        

        # Parse other kwargs. This allows for the xrf_fly specific defaults
        if 'min_roi_num' in scan_kwargs:
            min_roi_num = scan_kwargs.pop('min_roi_num')
        if 'max_roi_num' in scan_kwargs:
            max_roi_num = scan_kwargs.pop('max_roi_num') 
        if 'ignore_det_rois' in scan_kwargs:
            ignore_det_rois = scan_kwargs.pop('ignore_det_rois')       

        # Find rois
        # Do not even try any processing of failed scans. Too many potential failure points
        if (scan_data['exit_status'] == 'success'
            and 'stream0' in bs_run):

            # Load summed data. End above max SRX energy
            xrf_sum = np.sum(bs_run['stream0']['data']['xs_fluor'][..., :2500], axis=(0, 1, 2)).astype(np.float32)
            energy = np.arange(0, len(xrf_sum)) * 10

            # Load scaler for later
            sclr = bs_run['stream0']['data']['i0'][:].astype(np.float32)

            roi_dets = bs_run.start['scan']['detectors']
            roi_dets = [det for det in roi_dets if det not in ignore_det_rois]

            # Identify peaks and determine ROIS
            # May replace with saved ROIS later
            if 'xs' in roi_dets:
                (rois,
                rois_labels,
                peak_energies,
                peak_labels) = _find_xrf_rois(xrf_sum,
                                              energy,
                                              scan_data['energy'],
                                              min_roi_num=min_roi_num,
                                              max_roi_num=max_roi_num,
                                              scan_kwargs=scan_kwargs)
                
                print(f'{len(rois)=}')
            else:
                rois, rois_labels = [], []
            
            # Make a consideration for area detectors. Assume that if they were added, their signal is a useful roi
            # num_area_dets = sum([det in ['merlin', 'dexela'] for det in bs_runs.start['scan']['detectors']])
            dets_added = 0
            # Will add merlin first and only merlin if max_roi_num is 1
            if ('merlin' in roi_dets
                and dets_added < max_roi_num):
                rois.insert(0, 'merlin')
                rois_labels.insert(0, 'merlin')
                dets_added += 1
            if ('dexela' in roi_dets
                and dets_added < max_roi_num):
                rois.insert(0, 'dexela')
                rois_labels.insert(0, 'dexela')
                dets_added += 1

        else:
            rois = []

        print(f'{len(rois)=}')

        # Check for weird results
        if len(rois) == 0 and min_roi_num > 0:
            warn_str = ("WARNING: Could not find any interesting and "
                        + "significant ROIs for scan "
                        + f"{scan_data['scan_id']}.")
            print(warn_str)
            
        # Determine if a new page needs to be added
        max_height = 55 # Height of reference data
        num_images = np.min([len(rois), max_roi_num])
        # 13 for top banner, 1.5 for end, 1 for spacing
        space_needed = (13 + 1.5 + ((max_height + 1) * np.ceil(1 + (num_images - 1) / 3)))
        space_available = self.eph - self.y - self.b_margin
        print(f'Space needed for cell: {space_needed}')
        print(f'Space available: {space_available}')
        if space_available < space_needed:
            self.add_page()
        
        # Add base scan information
        self.add_BASE_SCAN(scan_data, scan_kwargs)
        
        # Load more useful metadata specific to XRF_FLY
        scan = bs_run.start['scan']

        # Compile information
        table_labels = ['Scan Input', 'Motors', 'Detectors', 'Pixel Dwell', 'Map Shape', 'Map Size', 'Step Sizes']
        args = scan['scan_input']
        print(f'{args=}')
        arg_str = ',\n'.join([', '.join([str(v) for v in args[:3]]), ', '.join([str(v) for v in args[3:6]])])
        exts = [args[1] - args[0], args[4] - args[3]]
        nums = [args[2], args[5]]
        steps = [ext / (num - 1) if num != 1 else 0 for ext, num in zip(exts, nums)]
        full_motors = [scan['fast_axis']['motor_name'], scan['slow_axis']['motor_name']]
        motors = [motor[11:] for motor in full_motors]
        motor_units = [scan['fast_axis']['units'], scan['slow_axis']['units']]

        table_values = [(f"{np.round(args[0], 3)}, {np.round(args[1], 3)}, {int(args[2])},"
                                + f"\n{np.round(args[3], 3)}, {np.round(args[4], 3)}, {int(args[2])}"),
                        ', '.join([str(motor) for motor in motors]),
                        ', '.join([str(v) for v in scan['detectors']]),
                        f"{scan['dwell']} sec",
                        ', '.join([str(int(v)) for v in scan['shape']]),
                        f"{np.round(exts[0], 3)} {motor_units[0]}, {np.round(exts[1], 3)} {motor_units[1]}",
                        f"{np.round(steps[0], 3)} {motor_units[0]}, {np.round(steps[1], 3)} {motor_units[1]}"
                        ]

        # Scan-specific data
        reset_y = self.y
        self.set_xy(self.x + 2.5, self.y)
        self.set_font(style='B', size=10)
        self.cell(h=5, new_x='LEFT', new_y='NEXT', text=f"Scan Data")
        self.set_font(size=9)
        l_margin = self.l_margin 
        self.l_margin = self.x # Temp move to set table location. Probably bad
        # print(f'Before table {self.x=}, {self.y=}')
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=4.5,
                col_widths=(19, 31),
                width=50,
                align='L',
                text_align='LEFT'
                ) as table:
            for i in range(len(table_labels)):
                row = table.row()
                row.cell(table_labels[i])
                row.cell(table_values[i])
            table_width = table._width
        self.set_xy(self.x + table_width, reset_y)
        self.l_margin = l_margin # Reset left margin
        
        max_width = self.epw - self.r_margin - self.x - 1.5
        if len(rois) > 1:
            max_width = np.min([max_width, self.epw / 3])

        # Check if inputs should be transposed for plotting
        scan_directions = []
        for motor in motors:
            if 'x' in motor:
                scan_directions.append('x')
            elif 'y' in motor:
                scan_directions.append('y')
            else:
                scan_directions.append(None)
        
        TRANSPOSED = False
        if (scan_directions[0] == 'y'
            and scan_directions[1] == 'x'):
            full_motors = full_motors[::-1]
            motor_units = motor_units[::-1]
            steps = steps[::-1]
            args = [*args[3:6], *args[:3], *args[6:]]
            TRANSPOSED = True

        img_ind = 0
        for roi_ind in range(len(rois)):
            if img_ind + 1 > max_roi_num:
                break
            print(f'ROI index {roi_ind} abscissa is {self.x}, {self.y}')
            print(f'Plotting image index {img_ind}')

            # XRF
            if isinstance(rois[roi_ind], slice):
                # Load data around ROI
                data = np.sum(bs_run['stream0']['data']['xs_fluor'][..., rois[roi_ind]], axis=(-2, -1,), dtype=np.float32)
                data /= sclr
                en_int = energy[rois[roi_ind]]
                roi_str = f"{rois_labels[roi_ind]}: {int(en_int[0])} - {int(en_int[-1])} eV"
            # XRD or DPC or other area detector techniques
            elif rois[roi_ind] in ['merlin', 'dexela']:
                # TODO: Search for dark-field internally
                dark = None

                # Check for data
                if f'{rois[roi_ind]}_image' not in bs_run['stream0']['data']:
                    warn_str = ("WARNING: Key not in stream0 for "
                                + f"{rois[roi_ind]} data from scan "
                                + f"{scan_data['scan_id']}. Proceding "
                                + "without changes.")
                    print(warn_str)
                    continue
                # Load data
                try:
                    # raise
                    # Pseudo binning. Take at max 400 pixels from either image axis.
                    # Loading data should not be improved due to chunking, but processing should be faster
                    data_shape = bs_run['stream0']['data'][f'{rois[roi_ind]}_image'].shape
                    data_slicing = tuple([slice(None), slice(None)]
                                         + [slice(None, None, int(s / 350) + 1) for s in data_shape[-2:]])

                    data = np.empty((data_shape[0],
                                     data_shape[1],
                                     int(((data_shape[2] - 1) / data_slicing[2].step) + 1),
                                     int(((data_shape[3] - 1) / data_slicing[3].step) + 1)))

                    for index in range(np.prod(data_shape[:2])):
                        indices = np.unravel_index(index, data_shape[:2])

                        data[indices] = bs_run['stream0']['data'][f'{rois[roi_ind]}_image'][*indices, *data_slicing[-2:]].astype(np.float32)


                    # Load data
                    data = bs_run['stream0']['data'][f'{rois[roi_ind]}_image'][data_slicing].astype(np.float32)
                    if dark is None:
                        dark = np.min(data, axis=(0, 1)) # approximate
                    data -= dark
                    data = median_filter(data, size=(1, 1, 2, 2)) # denoise
                    data = np.max(data, axis=(-2, -1))
                    data /= sclr
                    
                    # Rescale data
                    data -= np.min(data)
                    data /= np.max(data)
                    data *= 100

                    # data = np.log(data)
                    roi_str = f"{rois_labels[roi_ind]}_max"
                except Exception as e:
                    err_str = (f"{e}: Error loading {rois[roi_ind]} "
                                + f"from scan {scan_data['scan_id']}. "
                                + f"Proceding without changes.")
                    print(err_str)
                    continue
            else:
                warn_str = ("WARNING: Unknown ROI encountered of type "
                            + f"{type(rois[roi_ind])} for scan "
                            + f"{scan_data['scan_id']}. Only slices "
                            + "and strings are accepted for XRF_FLY.")
                print(warn_str)
                continue

            # Parsing data shapes
            start_shape = tuple([int(s) for s in scan['shape']])[::-1]
            # 1D data is fine
            print(f'{args=}')
            if 1 in data.shape:
                pass
            else:
                # Fly in y, transponsed data
                if TRANSPOSED:
                    data = data.T
                    start_shape = start_shape[::-1]
                    # Is it complete?
                    if start_shape != data.shape:
                        print(f'{start_shape=}')
                        print(f'{data.shape=}')
                        full_data = np.zeros(start_shape)
                        full_data[:] = np.nan
                        for i in range(data.shape[1]):
                            full_data[:, i] = data[:, i]
                        data = full_data
                else: # Fly in x
                    # Is the data complete?
                    if start_shape != data.shape:
                        print(f'{start_shape=}')
                        print(f'{data.shape=}')
                        full_data = np.zeros(start_shape)
                        full_data[:] = np.nan
                        for i in range(data.shape[0]):
                            full_data[i] = data[i]
                        data = full_data
            

            # Generate plot
            fig_ratio = max_width / max_height
            figsize = [max_width / 15, max_height / 15]
            fig, ax = plt.subplots(figsize=figsize, layout='tight', dpi=200)
            fontsize = 12
            if 1 in data.shape: # Plot plot
                if steps[0] != 0:
                    ax.plot(np.linspace(*args[:2], int(args[2])), data.squeeze())
                else:
                    ax.plot(np.linspace(*args[3:5], int(args[5])), data.squeeze())
                ax.tick_params(labelleft=False)
                ax.set_ylabel('Normalized Intensity [a.u.]', fontsize=fontsize)
            else: # Plot image
                extent = [args[0] - steps[0] / 2,
                          args[1] + steps[0] / 2,
                          args[4] + steps[1] / 2,
                          args[3] - steps[1] / 2]
                im = ax.imshow(data, extent=extent)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=0.1, pad=0.1)
                cbar = fig.colorbar(im, cax=cax)
                cbar.formatter.set_powerlimits((-3, 4))
                cbar.ax.tick_params(labelsize=fontsize)
                ax.set_aspect('equal')
                ax.set_ylabel(f"{full_motors[1]} [{motor_units[1]}]", fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.set_xlabel(f"{full_motors[0]} [{motor_units[0]}]", fontsize=fontsize)
            ax.set_title(f"Scan {bs_run.start['scan_id']}\n{roi_str}",
                         fontsize=fontsize,
                         pad=15)

            # fig.savefig(f'/nsls2/users/emusterma/Documents/scan_reports/figure_{roi_ind}.png')

            # Converting Figure to an image:
            canvas = FigureCanvas(fig)
            canvas.draw()
            pdf_img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
            plt.close(fig)

            # Rescale plot image. Maximize along one dimension maintaining aspect ratio
            width_at_max_height = (pdf_img.width / pdf_img.height) * max_height
            if width_at_max_height > max_width:
                img_height = max_width * (pdf_img.height / pdf_img.width)
                img_width = max_width
            else:
                img_height = max_height
                img_width = (pdf_img.width / pdf_img.height) * img_height
            
            # With tables
            if img_ind == 0:
                # print('Plotting first index') 
                img_x = self.x + ((self.w - self.r_margin) - self.x - img_width) / 2
                self.image(pdf_img, x=img_x, h=img_height, keep_aspect_ratio=True)
                self.set_xy(self.l_margin, reset_y + max_height + 1) # Extra 1 to prevent overlap
                reset_y = self.y

            # Left of three
            elif (img_ind - 1) % 3 == 0:
                # print('Plotting in location 0')
                img_x = self.l_margin
                self.image(pdf_img, x=img_x, h=img_height, keep_aspect_ratio=True)
                self.set_xy(self.l_margin, reset_y)

            # Middle of three
            elif (img_ind - 1) % 3 == 1:
                # print('Plotting in location 1')
                img_x = (self.w - img_width) / 2
                self.image(pdf_img, x=img_x, h=img_height, keep_aspect_ratio=True)
                self.set_xy(self.l_margin, reset_y)

            # Right of three    
            elif (img_ind - 1) % 3 == 2:
                # print('Plotting in location 2')
                img_x = self.w - self.r_margin - img_width
                self.image(pdf_img, x=img_x, h=img_height, keep_aspect_ratio=True)
                self.set_xy(self.l_margin, reset_y + max_height + 1) # add max height for next line
                reset_y = self.y
            
            # Move to next image index
            img_ind += 1
        
        # Cleanup
        if img_ind == 0 or (img_ind > 1 and (img_ind - 2) % 3 != 2):
            self.set_xy(self.l_margin, self.y + max_height + 1.5)
            print('first cleanup call')
        elif (img_ind - 1) % 3 != 2:
            self.set_xy(self.l_margin, self.y + 0.5)
            print('second cleanup call')



    def add_XAS_STEP(self,
                     bs_run,
                     scan_data,
                     scan_kwargs):
        """Add data specific to XAS_STEP scans"""

        # Determine if a new page needs to be added
        max_height = 55 # Height of reference data
        space_needed = (13 + 1.5 + max_height)
        space_available = self.eph - self.y - self.b_margin
        # print(f'Space needed for cell: {space_needed}')
        # print(f'Space available: {space_available}')
        if space_available < space_needed:
            self.add_page()
        
        # Add base scan information
        self.add_BASE_SCAN(scan_data, scan_kwargs)
        
        # Load more useful metadata specific to XRF_FLY
        scan = bs_run.start['scan']

        # all in eV
        table_labels = ['Energy Inputs', 'Energy Steps', 'Detectors', 'Pixel Dwell', 'Point Number', 'Range']
        scan_inputs = scan['scan_input'].split(', ')
        en_inputs = [float(en) for en in scan_inputs[0][1:-1].split(' ')]
        if en_inputs[0] < 1e3:
            en_inputs = [en * 1e3 for en in en_inputs]
        en_steps = scan_inputs[1][1:-1].split(' ') # should already be in eV
        dets = [det for det in bs_run.start['detectors'] if det != 'ring_current'] # ring current is unecessary
        nom_energy = [float(en) for en in scan['energy']]
        if nom_energy[0] < 1e3:
            nom_energy = [en * 1e3 for en in en_inputs]
        nom_energy = np.round(nom_energy, 1)
        en_range = (np.max(nom_energy) - np.min(nom_energy)) # in eV

        table_values = [',\n'.join([str(en) for en in en_inputs]),
                        ', '.join([str(en) for en in en_steps]),
                        ', '.join(dets),
                        f"{scan['dwell']} sec",
                        str(len(nom_energy)),
                        f"{en_range} eV"
                        ]

        reset_y = self.y
        self.set_xy(self.x + 1.5, self.y)
        self.set_font(style='B', size=10)
        self.cell(h=5, new_x='LEFT', new_y='NEXT', text=f"Scan Data")
        self.set_font(size=9)
        l_margin = self.l_margin 
        self.l_margin = self.x # Temp move to set table location. Probably bad
        with self.table(
                first_row_as_headings=False,
                line_height=4.5,
                col_widths=(19, 31),
                width=50,
                align='L',
                text_align='LEFT'
                ) as table:
            for i in range(len(table_labels)):
                row = table.row()
                row.cell(table_labels[i])
                row.cell(table_values[i])
            table_width = table._width
        self.set_xy(self.x + table_width, reset_y)
        self.l_margin = l_margin # Reset left margin
        max_width = self.w - self.r_margin - self.x - 1.5

        # Do not even try any processing of failed scans. Too many potential failure points
        if (scan_data['exit_status'] == 'success'
            and 'primary' in bs_run):
            
            en = bs_run['primary']['data']['energy_energy'][:].astype(np.float32)
            if en[0] < 1e3:
                en *= 1e3
            data = np.sum([bs_run['primary']['data'][f'xs_channel0{i + 1}_mcaroi01_total_rbv'][:] for i in range(8)], axis=0, dtype=np.float32)
            data /= bs_run['primary']['data']['sclr_i0'][:].astype(np.float32)
            edge_ind = np.argmax(np.gradient(data, en))
            el_edge = all_edges_names[np.argmin(np.abs(np.array(all_edges) - en[edge_ind]))]

            # Plot data
            fig, ax = plt.subplots(figsize=(max_width / 15, max_height / 15), tight_layout=True, dpi=200)
            fontsize = 12
            ax.plot(en, data)
            ax.scatter(en[edge_ind], data[edge_ind], marker='*', s=50, c='r')
            ax.tick_params(labelleft=False)
            ax.set_ylabel('Normalized Intensity [a.u.]', fontsize=fontsize)
            ax.set_xlabel('Energy [eV]', fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.set_title(f"Scan {bs_run.start['scan_id']}\n{el_edge} edge : {int(en[edge_ind])} eV", fontsize=fontsize)   

            # Converting Figure to an image:
            canvas = FigureCanvas(fig)
            canvas.draw()
            pdf_img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
            plt.close(fig)

            width_at_max_height = (pdf_img.width / pdf_img.height) * max_height
            if width_at_max_height > max_width:
                img_height = max_width * (pdf_img.height / pdf_img.width)
                img_width = max_width
            else:
                img_height = max_height
                img_width = (pdf_img.width / pdf_img.height) * img_height

            img_x = ((self.w - self.r_margin) - self.x - img_width) / 2
            self.image(pdf_img, x=img_x + self.x, h=img_height, keep_aspect_ratio=True)
        
        # Cleanup
        self.set_xy(self.l_margin, reset_y + max_height + 1.5)
    

    ### Extract information from tiled ###

    def _get_start_scan_data(self,
                             bs_run):
        """Get metadata from start document."""
        
        scan_start_data = {}

        start = bs_run.start
        stop = bs_run.stop

        if 'scan_id' in start:
            scan_start_data['scan_id'] = start['scan_id']
        
        if 'time' in start:
            scan_start_data['start_time'] = start['time']
        else:
            scan_start_data['start_time'] = None # Should never happen

        if 'scan' in start:
            if 'type' in start['scan']:
                scan_start_data['scan_type'] = start['scan']['type']
            else:
                scan_start_data['scan_type'] = 'UNKNOWN'

        if 'time' in stop:
            scan_start_data['stop_time'] = stop['time']
        else:
            scan_start_data['stop_time'] = None # Should never happen
        
        if 'exit_status' in stop:
            scan_start_data['exit_status'] = stop['exit_status']
        else:
            scan_start_data['exit_status'] = 'unknown'

        return scan_start_data
    

    def _get_baseline_scan_data(self,
                                bs_run):
        """Get reference data from baseline"""
        scan_base_data = {}

        baseline = bs_run['baseline']

        # Stage values!
        for key in ['x', 'y', 'z', 'topx', 'topz', 'th', 'sx', 'sy', 'sz']:
            full_key = f'nano_stage_{key}'
            if f'{full_key}_user_setpoint' in baseline['data']:
                scan_base_data[key] = baseline['data'][f'{full_key}_user_setpoint'][0]
            else:
                scan_base_data[key] = None
            if f'{full_key}_motor_egu' in baseline['config']['nano_stage']:
                unit = baseline['config']['nano_stage'][f'{full_key}_motor_egu'][0]
                # if unit == 'um':
                #     unit = 'μm'
                scan_base_data[f'{key}_units'] = unit   
            else:
                scan_base_data[f'{key}_units'] = None

        # Check th units:
        if scan_base_data['th_units'] == 'mdeg':
            scan_base_data['th'] = scan_base_data['th'] * 1e3
            scan_base_data['th_units'] = 'deg'
        
        # Energy values!
        if 'energy_energy_setpoint' in baseline['data']:
            en = baseline['data']['energy_energy_setpoint'][0]
            if en < 1e3:
                en *= 1e3
            scan_base_data['energy'] = en
            scan_base_data['energy_units'] = 'eV'
        else:
            scan_base_data['energy'] = None
            scan_base_data['energy_units'] = None
        
        # SDD position value!
        # Approximate until we get actual distance pv
        if 'nano_det_x' in baseline['data']:
            scan_base_data['sdd_x'] = baseline['data']['nano_det_x'][0]
            scan_base_data['sdd_x_units'] = baseline['config']['nano_det']['nano_det_x_motor_egu'][0]
        else:
            scan_base_data['sdd_x'] = None
            scan_base_data['sdd_x_units'] = None

        return scan_base_data
    
    
    def get_proposal_scan_data(self,
                               scan_id):
        """Get proposal information for header."""

        start = c[int(scan_id)].start
        exp_md = {}

        for key in ['proposal_id', 'title', 'pi_name']:
            if key in start['proposal']:
                exp_md[key] = start['proposal'][key]
            else:
                exp_md[key] = None
        
        UPDATED_EXP_MD = False
        for key, value in exp_md.items():
            if key not in self.exp_md:
                self.exp_md[key] = value
            elif self.exp_md[key] != value:
                UPDATED_EXP_MD = True
        
        return UPDATED_EXP_MD








scan_report = SRXScanPDF()
# scan_report.exp_md = {'proposal_number' : 315950,
#                       'proposal_title' : 'SRX Beamline Commissioning',
#                       'proposal_PI' : 'Kiss'}
#scan_report.t_margin = 20

# scan_report.add_scan_range(164503, 164623, include_peakups=False, include_failures=False, snr_cutoff=100)
# scan_report.add_scan_range(164528, 164530, include_peakups=False, include_failures=False, snr_cutoff=100)
# scan_report.add_scan_range(164503, 164505)
scan_report.add_scan_range(166789,
                           166791,
                           include_peakups=False,
                           include_failures=False,
                           max_roi_num=4,
                           ignore_det_rois=['dexela'])
# scan_report.add_scan(162006, include_peakups=False, include_failures=False, snr_cutoff=100)
# scan_report.add_scan(162021, include_peakups=False, include_failures=False, snr_cutoff=100)

# scan_report.add_scan(164505)
# scan_report.add_scan(164507)
# scan_report.add_scan(164535)
# scan_report.add_scan(164540)

scan_report.output('pass-316224_cycle_2025-1.pdf')