import numpy as np
import time as ttime
from tqdm import tqdm
import os
import io
import json
import dask.array as da

from fpdf import FPDF
from PyPDF2 import PdfReader, PdfWriter, PdfMerger

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
from scipy.signal import find_peaks
import skbeam.core.constants.xrf as xrfC
from itertools import combinations_with_replacement
from scipy.ndimage import median_filter
from scipy.stats import mode

from xrdmaptools.io.db_io import (
    manual_load_data,
    load_step_rc_data
)

from tiled.client import from_profile
from tiled.queries import Key

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
# edges = ['k', 'l3']
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
                   energy_tolerance=60, # What is good value for this? About half energy resolution right now.
                   esc_en=1740,
                   snr_cutoff=100,
                   scan_kwargs={},
                   verbose=False):

        # Parse out scan_kwargs
        if 'specific_elements' in scan_kwargs:
            specific_elements = scan_kwargs.pop('specific_elements')
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
        
        # Verbosity
        if verbose:
            print('Finding XRF ROI information.')
            print(f'{min_roi_num=}')
            print(f'{max_roi_num=}')
            print(f'{specific_elements=}')

        # Parse some inputs
        if min_roi_num > max_roi_num:
            max_roi_num = min_roi_num
        
        # Do not bother if no rois requested
        if max_roi_num == 0:
            return [], [], [], []

        # Process specified elements before anything else
        found_elements, specific_lines  = [], []
        num_interesting_rois = 0
        if specific_elements is not None:
            for el in specific_elements:
                # print(f'{el} in specific elements')
                if (isinstance(el, str)
                    and el.split('_')[0].capitalize() in possible_elements):
                    # Add check to see if the element name has been repeated
                    # print(f'{el} added to found elements')
                    found_elements.append(xrfC.XrfElement(el.split('_')[0].capitalize()))
                    specific_lines.append(el)
                    num_interesting_rois += 1

        # Convert energy to eV
        if energy[int((len(energy) - 1) / 2)] < 1e3:
            energy = np.array(energy) * 1e3
        if incident_energy < 1e3:
            incident_energy *= 1e3

        # Get energy step size
        en_step = np.mean(np.diff(energy), dtype=int)

        # Only try to find rois if the max_roi_num is not satisfied by specific_elements inputs
        if num_interesting_rois < max_roi_num:
            # Do not modify the original data
            xrf = xrf.copy()

            # Crop XRF and energy to reasonable limits (Assuming 10 eV steps)
            # No peaks below 1000 eV or above 85% of incident energy
            en_range = slice(int(1e3 / en_step), int(0.9 * incident_energy / en_step))
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
                    else: # Unlikely
                        found_elements.append(int(peak_en))
                        peak_labels.append('Unknown')
                        num_interesting_rois += 1

        # Generate new ROIS
        rois, roi_labels = [], []
        for el in found_elements:
            # Add specified lines first
            if isinstance(el, xrfC.XrfElement):
                if el.sym in [line.split('_')[0] for line in specific_lines]:
                    line = specific_lines[[line.split('_')[0] for line in specific_lines].index(el.sym)].split('_')[-1]
                    line_en = el.emission_line[line] * 1e3
                    rois.append(slice(int((line_en / en_step) - (100 / en_step)), int((line_en / en_step) + (100 / en_step))))
                    roi_labels.append(f'{el.sym}_{line}')
                    if verbose:
                        print(f'Specified element ({el.sym}) with line ({line}) added.')
                    continue
                
                # Ignore argon mostly
                elif el.sym in boring_elements:
                    if verbose:
                        print(f'Found element {el.sym}, but it is too boring to generate ROI!') 
                    continue            

            elif isinstance(el, int):
                if verbose:
                    print(f'Found unknown ROI around {el} eV.')
                rois.append(slice(int((el / en_step) - (100 / en_step)), int((el / en_step) + (100 / en_step))))
                roi_labels.append('Unknown')
                continue
            
            # Something bad happened
            else:
                print('WARNING: Weird element found in XRF ROIs. Moving on, but something is very wrong.')
            

            if verbose:
                print(f'Found element {el.sym}! Generating ROI around highest yield fluorescence line.')    
            # Slice major lines
            for line in roi_lines:
                line_en = el.emission_line[line] * 1e3
                if 1e3 < line_en < incident_energy:
                    if verbose:
                        print(f'Highest yield fluorescence line for {el.sym} is {line}.')
                    rois.append(slice(int((line_en / en_step) - (100 / en_step)), int((line_en / en_step) + (100 / en_step))))
                    roi_labels.append(f'{el.sym}_{line}')
                    break
            else:
                if verbose:
                    print(f'No fluorescence line found for {el.sym} with an incident energy of {incident_energy} eV!')
                    print(f'Cannot generate ROI for {el.sym}!')

        return rois, roi_labels


class SRXScanPDF(FPDF):

    def __init__(self, font_style='helvetica', verbose=False):
        FPDF.__init__(self)
        self.set_auto_page_break(False)
        self.set_font(font_style)

        # Custom
        self.exp_md = {}         
        self._disable_header = False
        self._disable_footer = False
        self._appended_pages = 0
        self._verbose = verbose

        # Formatting Variables
        # Spacing in mm
        # Heights
        self._gap_height = 1.5 # Extra space added for gaps
        self._banner_height = 12.5 # For banner information
        self._max_height = 55 # General cell height for tables and figures
        self._offset_height = 1 # Gap between header and table and between figures
        # General widths
        self._gap_width = 1.5

        # Fonts
        self._banner_font_size = 9.5
        self._table_header_font_size = 9
        self._table_font_size = 8
        
        # Table formatting
        # CAUTION: these values are related to the above spacing variables
        self._banner_table_cell_height = 4.75
        self._table_header_height = 4
        self._table_cell_height = 3.9
        # Widths
        self._base_table_width = 37
        self._base_table_cols = (17, 20)
        self._scan_table_width = 50
        self._scan_table_cols_xrf = (18, 32) # XRF_FLY
        self._scan_table_cols_xas = (20, 30) # XAS_STEP, XAS_FLY eventually
        self._scan_table_cols_rc = (18, 32) # ENERGY_RC, ANGLE_RC
        
        # Derived
        self._max_plot_width = (self.epw
                                - self._base_table_width
                                - self._scan_table_width
                                - (2 * self._gap_width))


    def header(self):
        if self.disable_header:
            return

        self.set_font(size=11)

        self.image('/nsls2/users/emusterma/Documents/Repositories/srx-scan-reports/data/srx_logo.png', h=15, keep_aspect_ratio=True)
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
                if len(title_text) + len(word) + 4 < 80:
                    title_text += f' {word}'
                else:
                    title_text += '...'
                    break
            table.row().cell(f"Proposal Title : {title_text}")

        self.set_line_width(0.5)
        self.set_draw_color(0, 0, 0)
        # Hard-coded gap of 1.5 mm. May be worth assigning to self._gap
        self.line(x1=self.l_margin, y1=self.y + self._gap_width, x2=self.epw + self.l_margin, y2=self.y + self._gap_width)

        post_header_abscissa = (self.x, self.y)

        # Render footer with header to track page generation
        self.disable_footer = False
        self.footer()
        self.disable_footer = True

        self.set_xy(*post_header_abscissa)


    def footer(self):
        if self.disable_footer:
            return

        self.set_xy(self.x, -15)
        self.set_font('helvetica', size=10)
        self.cell(0, 10, f'Page {self.total_pages()}', align='C')
        self.cell(0, 10, f'Generated on {ttime.ctime()}', align='R')


    @property
    def disable_header(self):
        return self._disable_header

    @disable_header.setter
    def disable_header(self, val):
        if not isinstance(val, bool):
            raise TypeError('Disable marginals attribute must be boolean.')
        self._disable_header = val
    

    @property
    def disable_footer(self):
        return self._disable_footer

    @disable_footer.setter
    def disable_footer(self, val):
        if not isinstance(val, bool):
            raise TypeError('Disable marginals attribute must be boolean.')
        self._disable_footer = val
    

    def total_pages(self):
        # Total current and appended pages
        return self.page_no() + self._appended_pages

    
    def add_page(self,
                 *args,
                 disable_header=False,
                 disable_footer=True,
                 **kwargs):
        
        # Add header/footer by default
        self.disable_header = disable_header
        self.disable_footer = disable_footer
        super().add_page(*args, **kwargs)
        self.disable_header = True
        self.disable_footer = True
    

    def request_cell_space(self,
                           num_images=None,
                           space_needed=None):
        # Check remaining space on page and add new page if insufficient
        
        space_available = self.eph - self.y - self.b_margin
        if space_needed is None:
            if num_images is None:
                raise ValueError('Must define space_needed or num_images!')
            space_needed = (self._banner_height
                            + self._gap_height
                            + self._offset_height
                            + ((self._max_height + self._offset_height) * np.ceil(1 + (num_images - 1) / 3))) # a touch awful...
        
        if self._verbose:
            print(f'Space needed for cell: {space_needed}')
            print(f'Space available: {space_available}')

        if space_available < space_needed:
            self.add_page()
        

    def add_scan(self,
                 scan_id,
                 include_optimizers=True,
                 include_unknowns=True,
                 include_failures=True,
                 **kwargs):
        """Add a scan from a scan_id"""

        scan_kwargs = kwargs
              
        # Add first page if empty
        if self.exp_md == {}:
            self.get_proposal_scan_data(scan_id)
            self.add_page()

        bs_run = c[int(scan_id)]
        # Extract metadata from start
        scan_data = self.get_start_scan_data(bs_run)

        # Skip failed scans if not including failures
        if (not include_failures
            and scan_data['exit_status'] != 'success'):
            return

        # Extract reference data from baseline
        scan_data.update(self.get_baseline_scan_data(bs_run))        

        # Build report entry
        if scan_data['scan_type'] == 'XRF_FLY':
            start_y = self.y
            self.add_XRF_FLY(bs_run, scan_data, scan_kwargs)
            end_y = self.y
            if self._verbose:
                print(f'Cell took {end_y - start_y} mm.')

        elif scan_data['scan_type'] == 'XAS_STEP':
            start_y = self.y
            self.add_XAS_STEP(bs_run, scan_data, scan_kwargs)
            end_y = self.y
            if self._verbose:
                print(f'Cell took {end_y - start_y} mm.')

        elif scan_data['scan_type'] == 'XAS_FLY':
            start_y = self.y
            self.add_XAS_FLY(bs_run, scan_data, scan_kwargs)
            end_y = self.y
            if self._verbose:
                print(f'Cell took {end_y - start_y} mm.')

        elif scan_data['scan_type'] == 'XRF_STEP':
            start_y = self.y
            self.add_XRF_STEP(bs_run, scan_data, scan_kwargs)
            end_y = self.y
            if self._verbose:
                print(f'Cell took {end_y - start_y} mm.')

        elif scan_data['scan_type'] == 'ANGLE_RC':
            start_y = self.y
            self.add_ANGLE_RC(bs_run, scan_data, scan_kwargs)
            end_y = self.y
            if self._verbose:
                print(f'Cell took {end_y - start_y} mm.')

        elif scan_data['scan_type'] == 'ENERGY_RC':
            start_y = self.y
            self.add_ENERGY_RC(bs_run, scan_data, scan_kwargs)
            end_y = self.y
            if self._verbose:
                print(f'Cell took {end_y - start_y} mm.')
        
        elif scan_data['scan_type'] == 'STATIC_XRD':
            start_y = self.y
            self.add_STATIC_XRD(bs_run, scan_data, scan_kwargs)
            end_y = self.y
            if self._verbose:
                print(f'Cell took {end_y - start_y} mm.')

        elif scan_data['scan_type'] in ['PEAKUP', 'OPTIMIZE_SCALERS']:
            if include_optimizers:
                start_y = self.y
                self.request_cell_space(space_needed=(self._banner_height
                                                      + self._gap_height))
                self.add_BASE_SCAN(scan_data, scan_kwargs, add_space=True)
                end_y = self.y
                if self._verbose:
                    print(f'Cell took {end_y - start_y} mm.')
            else:
                return

        elif include_unknowns:
            warn_str = (f"WARNING: Scan {scan_id} of type {scan_data['scan_type']} "
                        + "not yet supported for SRX scan reports.")
            print(warn_str)
            start_y = self.y
            self.request_cell_space(space_needed=(self._banner_height
                                                  + self._max_height
                                                  + self._gap_height
                                                  + self._offset_height))
            self.add_BASE_SCAN(scan_data, scan_kwargs, add_space=True)
            end_y = self.y
            if self._verbose:
                print(f'Unknown took {end_y - start_y} mm.')

        else:
            return

    
    def add_BASE_SCAN(self,
                      scan_data,
                      scan_kwargs,
                      add_space=False):
        """Add basic data for all scans"""

        if self._verbose:
            print('Adding BASE_SCAN...')

        # Generate starting line
        self.set_line_width(0.5)
        self.set_draw_color(0, 0, 0)
        self.set_xy(self.x, self.y + self._gap_height)
        self.line(x1=self.l_margin, y1=self.y, x2=self.epw + self.l_margin, y2=self.y)
        if scan_data['stop_time'] is not None:
            stop_time = ttime.strftime('%b %d %H:%M:%S', ttime.localtime(scan_data['stop_time']))
            duration = ttime.strftime('%H:%M:%S', ttime.gmtime(scan_data['stop_time'] - scan_data['start_time']))
        else:
            stop_time = 'unknown'
            duration = 'unknown'

        scan_labels = ['ID', 'Type', 'Status', 'Start', 'Stop', 'Duration']
        if len(scan_data['scan_type']) <= 12:
            scan_type_str = scan_data['scan_type']
        else:
            scan_type_str = f"{scan_data['scan_type'][:12]}..."
        scan_values = [str(scan_data['scan_id']),
                       scan_type_str,
                       scan_data['exit_status'].upper(),
                       ttime.strftime('%b %d %H:%M:%S', ttime.localtime(scan_data['start_time'])),
                       stop_time,
                       duration]
        
        # General scan data
        self.set_xy(self.x, self.y + self._gap_height)
        self.set_line_width(0.1)
        self.set_font(size=self._banner_font_size)
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=self._banner_table_cell_height,
                align='L',
                ) as table:
            row = table.row()
            for i in range(len(scan_labels)):
                if i == 0:
                    self.set_font(style='B', size=self._banner_font_size)
                row.cell(f'{scan_labels[i]}')
                if i == 0:
                    self.set_font(style='', size=self._banner_font_size)
            row = table.row()
            for i in range(len(scan_labels)):
                if i == 0:
                    self.set_font(style='B', size=self._banner_font_size)
                row.cell(str(scan_values[i]))
                if i == 0:
                    self.set_font(size=self._banner_font_size)
        
        # Do not add reference positions for failed scans or optimizers?
        if (scan_data['scan_type'] in ['PEAKUP', 'OPTIMIZE_SCALERS']):
            if add_space:
                self.set_xy(self.l_margin, self.y + self._gap_height)
            return

        # Main table components
        ref_labels = ['Energy', 'Coarse X', 'Coarse Y', 'Coarse Z', 'Top X', 'Top Z', 'Theta', 'Scanner X', 'Scanner Y', 'Scanner Z']
        ref_values = [scan_data[key] for key in ['energy', 'x', 'y', 'z', 'topx', 'topz', 'th', 'sx', 'sy', 'sz']]
        ref_units = [scan_data[f'{key}_units'] for key in ['energy', 'x', 'y', 'z', 'topx', 'topz', 'th', 'sx', 'sy', 'sz']]
        ref_precision = [0, 1, 1, 1, 1, 1, 3, 3, 3, 3]

        # Add SDD information
        # This conditional is for backwards compatibility before sdd dist was recorded in baseline
        if (scan_data['sdd_dist'] is None
            and scan_data['sdd_x'] is not None):
            ref_labels.append('SDD X')
            ref_values.append(scan_data['sdd_x'])
            ref_units.append(scan_data['sdd_x_units'])
            ref_precision.append(0)
        else:
            ref_labels.append('SDD Dist.')
            ref_values.append(scan_data['sdd_dist'])
            ref_units.append(scan_data['sdd_x_units']) # Should be the same
            ref_precision.append(0)

        # Parse attenuators:
        atten = {'Al' : 0, 'Si' : 0}
        for key, val in scan_data.items():
            if 'attenuators_' in key and val == 1:
                el, t = key[12:].split('_')
                atten[el] += float(t[:-2])
        atten_str = ''
        for key, val in atten.items():
            if val > 0:
                if atten_str != '':
                    atten_str += '\n'
                atten_str += f'{val:.0f} um {key}'
        if atten_str == '':
            atten_str = 'None'
        
        # Reference data
        self.set_xy(self.x, self.y + self._offset_height)
        reset_y = self.y
        self.set_font(style='B', size=self._table_header_font_size)
        self.cell(h=self._table_header_height, new_x='LMARGIN', new_y='NEXT', text=f"Reference Data")
        self.set_font(size=self._table_font_size)
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=self._table_cell_height,
                col_widths=self._base_table_cols,
                width=self._base_table_width,
                align='L'
                ) as table:
            for i in range(len(ref_labels)):
                row = table.row()
                row.cell(ref_labels[i])
                if ref_values[i] is not None:
                    row.cell(f'{np.round(ref_values[i], ref_precision[i]):.{ref_precision[i]}f}' + f' {ref_units[i]}')
                else:
                    row.cell('-')

            # Add attenuators
            row = table.row()
            row.cell('Total\nAttenuation')
            row.cell(atten_str)

            table_width = table._width
        
        if add_space:
            self.set_xy(self.l_margin, self.y + self._gap_height)
        else:
            self.set_xy(self.x + table_width, reset_y)
        
        if self._verbose:
            print('BASE_SCAN added!')


    def add_XRF_FLY(self,
                    bs_run,
                    scan_data,
                    scan_kwargs):
        """Add data specific to XRF_FLY scans"""

        if self._verbose:
            print('Adding XRF_FLY...')

        self._add_xrf_general(bs_run,
                              scan_data,
                              scan_kwargs,
                              scan_type='fly')


    def add_XRF_STEP(self,
                     bs_run,
                     scan_data,
                     scan_kwargs):
        """Add data specific to XRF_STEP scans"""

        if self._verbose:
            print('Adding XRF_STEP...')

        self._add_xrf_general(bs_run,
                              scan_data,
                              scan_kwargs,
                              scan_type='step')


    def _add_xrf_general(self,
                         bs_run,
                         scan_data,
                         scan_kwargs,
                         scan_type,
                         min_roi_num=1,
                         max_roi_num=10, # Hard capping for too many elements
                         scaler_rois=[],
                         ignore_det_rois=[],
                         colornorm='linear'):
        """Add data specific to XRF_FLY and XRF_STEP scans"""

        # Parse scan_type
        scan_type = scan_type.lower()
        if scan_type == 'fly':
            stream = 'stream0'
        elif scan_type == 'step':
            stream = 'primary'
        else:
            raise ValueError("xrf scan_type must be 'fly' or 'step'")

        # Parse other kwargs. This allows for the xrf_fly specific defaults
        if 'min_roi_num' in scan_kwargs:
            min_roi_num = scan_kwargs.pop('min_roi_num')
        if 'max_roi_num' in scan_kwargs:
            max_roi_num = scan_kwargs.pop('max_roi_num')
        if 'scaler_rois' in scan_kwargs:
            scaler_rois = scan_kwargs.pop('scaler_rois')
        if 'ignore_det_rois' in scan_kwargs:
            ignore_det_rois = scan_kwargs.pop('ignore_det_rois')
        if 'colornorm' in scan_kwargs:
            colornorm = scan_kwargs.pop('colornorm')              

        # Load more useful metadata specific to XRF mapping
        scan = bs_run.start['scan']

        # Generate table information
        table_labels = ['Scan Input', 'Motors', 'Detectors', 'Pixel Dwell', 'Map Shape', 'Map Size', 'Step Sizes']
        args = scan['scan_input']
        input_str = (f"{np.round(args[0], 3)}, {np.round(args[1], 3)}, {int(args[2])},"
                                + f"\n{np.round(args[3], 3)}, {np.round(args[4], 3)}, {int(args[5])}")
        full_motors = [scan['fast_axis']['motor_name'], scan['slow_axis']['motor_name']]
        motors = [motor[11:] for motor in full_motors]
        motor_units = [scan['fast_axis']['units'], scan['slow_axis']['units']]
        if 'dwell' in scan:
            dwell = scan['dwell']
        else:
            dwell = args[6]
        
        # Convert angle deg to mdeg
        if motors[0] == 'th':
            args[0] /= 1e3
            args[1] /= 1e3
            motor_units[0] = 'deg'
        elif motors[1] == 'th':
            args[3] /= 1e3
            args[4] /= 1e3
            motor_units[1] = 'deg'

        exts = [args[1] - args[0], args[4] - args[3]]
        if scan_type == 'fly':
            nums = [args[2], args[5]]
            steps = [ext / (num - 1) if num != 1 else 0 for ext, num in zip(exts, nums)]
        else:
            steps = [args[2], args[5]]
            nums = scan['shape']
        all_dets = [det for det in scan['detectors']]
        useful_dets = [det for det in all_dets if det not in ['ring_current', 'nano_vlm']]
        roi_dets = [det for det in useful_dets if det not in ignore_det_rois]        

        table_values = [input_str,
                        ', '.join([str(motor) for motor in motors]),
                        ', '.join(useful_dets),
                        f"{dwell} sec",
                        ', '.join([str(int(v)) for v in scan['shape']]),
                        f"{np.round(exts[0], 3)} {motor_units[0]}, {np.round(exts[1], 3)} {motor_units[1]}",
                        f"{np.round(steps[0], 3)} {motor_units[0]}, {np.round(steps[1], 3)} {motor_units[1]}"
                        ]

        # Find rois for plotting
        # Do not even try any processing of failed scans. Too many potential failure points
        rois, roi_labels = [], []
        if (scan_data['exit_status'] == 'success'
            and stream in bs_run):

            # Load scaler for nomalization later. May be reloaded if plotting scaler keys
            sclr_keys = ['i0', 'im']
            if scan_type == 'step':
                sclr_keys = [f'sclr_{key}' for key in sclr_keys]
            
            for sclr_key in sclr_keys:
                if not np.all(bs_run[stream]['data'][sclr_key][:] <= 0):
                    if scan_type == 'fly':
                        sclr = bs_run[stream]['data'][sclr_key][:].astype(np.float32)
                    else:
                        sclr = self._load_and_reshape_step_data(bs_run, sclr_key)
                    sclr[sclr < 1] = np.mean(sclr[sclr >= 1]) # Handles zeros
                    break
            else:
                warn_str = (f"WARNING: Issue encountered with scaler "
                            + f"for scan {bs_run.start['scan_id']}. "
                            + "Proceeding with unormalized data.")
                print(warn_str)
                sclr = np.ones(bs_run[stream]['data'][sclr_key].shape)

            # Add rois and roi_labels
            # Add vlm if acquired and not ignored
            if ('nano_vlm' in all_dets
                and not any(['vlm' in det for det in ignore_det_rois])
                and 'camera_snapshot' in bs_run
                and 'nano_vlm_image' in bs_run['camera_snapshot']['data']):
                rois.append('vlm')
                roi_labels.append('vlm')
            # Add scaler keys
            for scaler_roi in scaler_rois:
                # This check could happen sooner...
                if str(scaler_roi).lower() not in ['i0', 'im', 'it']:
                    warn_str = (f"WARNING: Scaler roi of {str(scaler_roi).lower()} "
                                + "not in accepted scalers of ['i0', 'im', 'it']."
                                + "\nSkipping scaler roi.")
                    print(warn_str)
                    continue
                rois.append(str(scaler_roi).lower())
                roi_labels.append(str(scaler_roi).lower())
            # Add area detectors
            if 'dexela' in roi_dets:
                rois.append('dexela')
                roi_labels.append('dexela')
            if 'merlin' in roi_dets:
                rois.append('merlin')
                roi_labels.append('merlin')
            # Add xs rois
            if ('xs' in roi_dets
                and len(rois) < max_roi_num):
                # Grab user-determined roi_information
                if isinstance(scan['detectors'], dict):
                    xs_det_rois = scan['detectors']['xs']
                    xs_rois = [xs_det_rois[f'roi{ind + 1}'] for ind in range(len(xs_det_rois))]
                    xs_rois = [roi for roi in xs_rois if roi != '']

                    if 'specific_elements' in scan_kwargs:
                        scan_kwargs['specific_elements'] = xs_rois + scan_kwargs['specific_elements']
                    else:
                        scan_kwargs['specific_elements'] = xs_rois
                
                # Load summed data
                if scan_type == 'fly':
                    try:
                        xrf_sum = np.sum(bs_run[stream]['data']['xs_fluor'][..., :7, :2500], axis=(0, 1, 2)).astype(np.float32)
                    except MemoryError:
                        # Dask version
                        print('WARNING: Dask invoked for sum XRF spectra!')
                        xrf_sum = da.asarray(bs_run[stream]['data']['xs_fluor'])[..., :7, :2500].sum(axis=(0, 1, 2)).compute().astype(np.float32)
                else:
                    xrf_sum = np.asarray([bs_run[stream]['data'][f'xs_channel0{ind + 1}_fluor'][..., :2500] for ind in range(7)]).sum(axis=(0, 1))
                    
                
                # Determine energy. Hard coded for current xpress3
                energy = np.arange(0, len(xrf_sum)) * 10                

                # Identify peaks and determine ROIS
                (xs_rois, xs_roi_labels) = _find_xrf_rois(xrf_sum,
                                                          energy,
                                                          scan_data['energy'],
                                                          min_roi_num=min_roi_num,
                                                          max_roi_num=max_roi_num,
                                                          scan_kwargs=scan_kwargs,
                                                          verbose=self._verbose)
                # This can extend rois beyond the max_roi_num
                rois.extend(xs_rois)
                roi_labels.extend(xs_roi_labels)

            # Check for weird results
            if len(rois) == 0 and min_roi_num > 0:
                warn_str = ("WARNING: Could not find any interesting and "
                            + "significant ROIs for scan "
                            + f"{scan_data['scan_id']}.")
                print(warn_str)
        
        else:
            rois = []
        
        # Generate images from rois
        images, max_widths = [], []
        if scan_data['exit_status'] == 'success':
            # Is mapped data transposed?
            if scan_type == 'fly':
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
            else:
                TRANSPOSED = False
            
            # Determine image sizes
            
            # Iterate through rois
            for roi, roi_label in zip(rois, roi_labels):
                if len(images) > max_roi_num:
                    break
                
                plot_func = None
                if len(rois) == 1:
                    # Single image can be a little bigger
                    roi_max_width = self._max_plot_width
                else:
                    roi_max_width = np.min([self._max_plot_width, self.epw / 3])

                # VLM
                if roi in ['vlm']:
                    fig = self._get_vlm_plot(bs_run,
                                            max_width=self._max_plot_width)
                    roi_max_width = self._max_plot_width # Only one that is bigger...
                # XRF
                elif isinstance(roi, slice):
                    plot_func = lambda *a, **k : self._get_xrf_map_plot(*a, energy=energy, **k)
                # XRD or DPC
                elif roi in ['merlin', 'dexela']:
                    plot_func = lambda *a, **k : self._get_ad_map_plot(*a, **k)
                # Scaler or XBIC
                elif rois[roi_ind] in [str(s).lower() for s in scaler_rois]:
                    plot_func = lambda *a, **k : self._get_sclr_map_plot(*a, **k)
                # Unknown
                else:
                    warn_str = ("WARNING: Unknown ROI encountered of type "
                                + f"{type(rois[roi_ind])} for scan "
                                + f"{scan_data['scan_id']}. Only slices "
                                + "and strings are accepted for XRF_FLY.")
                    print(warn_str)
                    fig = None
                
                if plot_func is not None:
                    fig = plot_func(bs_run,
                                    args,
                                    roi,
                                    roi_label,
                                    TRANSPOSED,
                                    sclr,
                                    sclr_key,
                                    full_motors,
                                    motor_units,
                                    max_width=roi_max_width,
                                    colornorm=colornorm
                                    )

                if fig is not None:
                    images.append(self._figure_to_image(fig))
                    max_widths.append(roi_max_width)
            
        # Determine if a new page needs to be added
        self.request_cell_space(num_images=len(images))
        
        # Add base scan information
        self.add_BASE_SCAN(scan_data, scan_kwargs)

        # Draw table
        reset_y = self.y
        self.set_xy(self.x + self._gap_width, self.y)
        self.set_font(style='B', size=self._table_header_font_size)
        self.cell(h=self._table_header_height, new_x='LEFT', new_y='NEXT', text=f"Scan Data")
        self.set_font(size=self._table_font_size)
        l_margin = self.l_margin 
        self.l_margin = self.x # Temp move to set table location. Probably bad
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=self._table_cell_height,
                col_widths=self._scan_table_cols_xrf,
                width=self._scan_table_width,
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
        
        # Draw images
        self.draw_images(images, max_widths, self._max_height)


    def add_XAS_STEP(self,
                     bs_run,
                     scan_data,
                     scan_kwargs):
        """Add data specific to XAS_STEP scans"""

        if self._verbose:
            print('Adding XAS_STEP...')

        self._add_xas_general(bs_run,
                              scan_data,
                              scan_kwargs,
                              scan_type='step')


    def add_XAS_FLY(self,
                    bs_run,
                    scan_data,
                    scan_kwargs):
        """Add data specific to XAS_STEP scans"""

        if self._verbose:
            print('Adding XAS_FLY...')

        self._add_xas_general(bs_run,
                              scan_data,
                              scan_kwargs,
                              scan_type='fly')

    
    def _add_xas_general(self,
                         bs_run,
                         scan_data,
                         scan_kwargs,
                         scan_type):
        """Add data specific to XAS_STEP and XAS_FLY scans"""

        # Parse scan_type
        scan_type = scan_type.lower()
        if scan_type not in ['fly', 'step']:
            raise ValueError("xas scan_type must be 'fly' or 'step'")
        
        # Load more useful metadata specific to XRF_FLY
        scan = bs_run.start['scan']

            # Parse detectors
        for d in [scan, bs_run.start]:
            if 'detectors' in d:
                all_dets = [det for det in d['detectors']]
                break
        else:
            all_dets = []
        useful_dets = [det for det in all_dets if det not in ['nano_vlm', 'ring_current']]

        # Generate scan table information
        if scan_type == 'step':
            table_labels = ['Energy Inputs', 'Energy Steps', 'Detectors', 'Point Dwell', 'Point Number', 'Range']
            scan_inputs = scan['scan_input'].split(', ')
            en_inputs = [float(en) for en in scan_inputs[0][1:-1].split(' ')]
            if en_inputs[0] < 1e3:
                en_inputs = [en * 1e3 for en in en_inputs]
            en_steps = scan_inputs[1][1:-1].split(' ') # should already be in eV
            en_range = np.max(en_inputs) - np.min(en_inputs)
            nom_energy = [float(en) for en in scan['energy']] # No better way to get number of points?
            en_start = en_inputs[0]
            en_end = en_inputs[-1]

            table_values = [',\n'.join([str(en) for en in en_inputs]),
                            ', '.join([str(en) for en in en_steps]),
                            ', '.join(useful_dets),
                            f"{scan['dwell']} sec",
                            str(len(nom_energy)),
                            f"{en_range} eV"
                            ]

        elif scan_type == 'fly':
            table_labels = ['Energy Start', 'Energy Stop', 'Energy Step', 'Detectors', 'Point Dwell', 'Point Number', 'Range', 'Harmonic']
            scan_inputs = scan['scan_input']
            for i in range(2):
                if scan_inputs[i] < 1e3:
                    scan_inputs[i] *=1e3
            en_start, en_end, en_step, dwell, num_points = scan_inputs
            en_range = np.max(en_end - en_start)

            table_values = [f"{en_start} eV",
                            f"{en_end} eV",
                            f"{en_step} eV",
                            ', '.join(useful_dets),
                            f"{dwell} sec",
                            f"{num_points}",
                            f"{en_range} eV",
                            f"{scan['harmonic']}"
                            ]
        
        # Generate images
        images, max_widths = [], []
        if scan_data['exit_status'] == 'success':
            # VLM
            if 'nano_vlm' in all_dets:
                fig = self._get_vlm_plot(bs_run,
                                        max_width=self._max_plot_width)
                if fig is not None:
                    images.append(self._figure_to_image(fig))
                    max_widths.append(self._max_plot_width)
            
            red_edges_mask = ((np.asarray(all_edges) > en_start)
                             & (np.asarray(all_edges) < en_end))
            red_edges_names = [name for b, name in zip(red_edges_mask, all_edges_names) if b]
            red_edges = [edge for b, edge in zip(red_edges_mask, all_edges) if b]
            en_step = 10 # Hard-coded for now

            # XAS      
            # if 'roi_names' in scan:
            #     roi_label = scan['roi_names'][int(scan['roi_num']) - 1]
            # elif ('detectors' in scan
            #       and isinstance(scan['detectors'], dict)):
            #     roi_label = scan['detectors']['xs']['roi1']
            # else:
            #     roi_label = 'Unknown'
            roi_label = None
            if 'roi_names' in scan:
                roi_labels = scan['roi_names']
            elif ('detectors' in scan
                  and isinstance(scan['detectors'], dict)):
                roi_labels = scan['detectors']['xs'].values()
            else:
                roi_labels = ['Unknown']
                roi_label = roi_labels[0]
            
            # Find best label. Should be first, but users can make mistakes
            if roi_label is None:
                if self._verbose:
                    print('Using ROI from scan metadata for XAS plotting.')
                roi_labels = [label for label in roi_labels if label != '']
                # Use first specified label with edge in energy range
                for label in roi_labels:
                    el, line = label.split('_')
                    for edge_name in red_edges_names:
                        if el == edge_name.split('_')[0]:
                            roi_label = edge_name
                            line_en = xrfC.XrfElement(el).emission_line[line] * 1e3
                            roi = slice(int((line_en / en_step) - (100 / en_step)),
                                        int((line_en / en_step) + (100 / en_step)))
                            break
                    else:
                        continue
                    break
                # Otherwise pull from XRF
                else:
                    if self._verbose:
                        print('Cannot find XAS ROI in scan metadata. Guessing for XRF spectrum.')
                    if (scan_type == 'fly' and any(['scan' in key for key in bs_run.keys()])):
                        stream_names = [key for key in bs_run.keys() if 'scan' in key]
                        xrf_sum = np.sum([bs_run[stream_names[0]]['data'][f'xs_id_mono_fly_channel0{ind + 1}'][..., :2500] for ind in range(7)], axis=(0, 1))
                    else:
                        xrf_sum = np.sum([bs_run['primary']['data'][f'xs_channel0{i + 1}_fluor'][..., :2500] for i in range(7)], axis=(0, 1))
                    
                    xs_roi = _find_xrf_rois(xrf_sum,
                                            np.arange(0, len(xrf_sum)) * en_step,
                                            en_end)
                    
                    for roi, roi_label in zip(*xs_roi):
                        el, line = roi_label.split('_')
                        for edge_name in red_edges_names:
                            if el == edge_name.split('_'):
                                break
                    # Give up
                    else:
                        if self._verbose:
                            print('Failed to find ROI from XRF spectrum. ROI fitting will be bad.')
                        if len(red_edges) > 0:
                            roi_label = red_edges_names[0]
                            roi = int(np.round(red_edges[0] / 1e3))
                        elif len(roi_labels) > 0:
                            roi_label = roi_labels[0]
                            el, line = roi_label.split('_')
                            line_en = xrfC.XrfElement(el).emission_line[line] * 1e3
                            roi = slice(int((line_en / en_step) - (100 / en_step)),
                                        int((line_en / en_step) + (100 / en_step)))
                        else:
                            # No idea, try everything...
                            roi_label = 'Total XRF'
                            roi = slice(20, 2500)

            # if roi_label != 'Unknown':
            #     el, line = roi_label.split('_')
            #     line_en = xrfC.XrfElement(el).emission_line[line] * 1e3
            #     en_step = 10 # Hard-coded for now
            #     roi = slice(int((line_en / en_step) - (100 / en_step)),
            #                 int((line_en / en_step) + (100 / en_step)))

            #     # Reduce possible edges...
            #     red_edges_mask = [el in name for name in all_edges_names]
            #     red_edges_names = [name for b, name in zip(red_edges_mask, all_edges_names) if b]
            #     red_edges = [edge for b, edge in zip(red_edges_mask, all_edges) if b]
            # else:
            #     red_edges_names = all_edges_names.copy()
            #     red_edges = all_edges.copy()

            # Load data
            # TODO: Allow for multiple edges with XAS_FLY?
            data = None
            if scan_type == 'step' and 'primary' in bs_run:
                en = bs_run['primary']['data']['energy_energy'][:].astype(np.float32)
                if en[0] < 1e3:
                    en *= 1e3
                if roi_label == roi_labels[0]:
                    data = np.sum([bs_run['primary']['data'][f'xs_channel0{i + 1}_mcaroi01_total_rbv'][:] for i in range(7)], axis=0, dtype=np.float32)
                else: 
                    data = np.sum([bs_run['primary']['data'][f'xs_channel0{i + 1}_fluor'][..., roi] for i in range(7)], axis=(0, -1))
                data /= bs_run['primary']['data']['sclr_i0'][:].astype(np.float32)
                edge_ind = np.argmax(np.gradient(data, en))
                el_edge = red_edges_names[np.argmin(np.abs(np.array(red_edges) - en[edge_ind]))]

                marker = np.asarray([[en[edge_ind], data[edge_ind]]])
                title = f"Scan {bs_run.start['scan_id']}\n{el_edge} edge : {int(en[edge_ind])} eV"
                data = [data]
                en = [en]
                labels = None
            
            elif (scan_type == 'fly' and any(['scan' in key for key in bs_run.keys()])):
                try:
                    stream_names = [key for key in bs_run.keys() if 'scan' in key]
                    en, data, marker, labels = [], [], [], []
                    for name in stream_names:
                        en_i = bs_run[name]['data']['energy'][:].astype(float)
                        data_i = np.sum([bs_run[name]['data'][f'xs_id_mono_fly_channel0{ind + 1}'][..., roi]
                                         for ind in range(7)], axis=(0, -1)).astype(float)
                        data_i /= bs_run[name]['data']['i0'][:].astype(float)
                        edge_ind = np.argmax(np.gradient(data_i, en_i))
                        el_edge = red_edges_names[np.argmin(np.abs(np.array(red_edges) - en_i[edge_ind]))]
                        marker.append((en_i[edge_ind], data_i[edge_ind]))

                        data.append(data_i)
                        en.append(en_i)
                    
                    labels = stream_names
                    # Title determined from last scan
                    title = f"Scan {bs_run.start['scan_id']}\n{el_edge} edge : {int(np.mean(marker, axis=0)[0])} eV"
                except Exception as e:
                    err_str = (f"{e}: Error loading {roi_label} data "
                                + f"from XRF_FLY scan {bs_run.start['scan_id']}. "
                                + f"proceeding without changes.")
                    print(err_str)
                    data = None

            # Plot_data
            if data is not None:
                fig = self._get_line_plot(
                        en,
                        data,
                        marker=marker,
                        title=title,
                        labels=labels,
                        y_label='Normalized Intensity [a.u.]',
                        x_label='Energy [eV]',
                        max_height=self._max_height,
                        max_width=self._max_plot_width)
                images.append(self._figure_to_image(fig))
                max_widths.append(self._max_plot_width)

        # Determine if a new page needs to be added
        self.request_cell_space(num_images=len(images))
        
        # Add base scan information
        self.add_BASE_SCAN(scan_data, scan_kwargs)

        # Draw table
        reset_y = self.y
        self.set_xy(self.x + self._gap_width, self.y)
        self.set_font(style='B', size=self._table_header_font_size)
        self.cell(h=self._table_header_height, new_x='LEFT', new_y='NEXT', text=f"Scan Data")
        self.set_font(size=self._table_font_size)
        l_margin = self.l_margin 
        self.l_margin = self.x # Temp move to set table location. Probably bad
        with self.table(
                first_row_as_headings=False,
                line_height=self._table_cell_height,
                col_widths=self._scan_table_cols_xas,
                width=self._scan_table_width,
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

        # Draw images
        self.draw_images(images, max_widths, self._max_height)


    def add_ENERGY_RC(self,
                      bs_run,
                      scan_data,
                      scan_kwargs):
        """Add data specific to ENERGY_RC scans"""

        if self._verbose:
            print('Adding ENERGY_RC...')

        self._add_step_rc(bs_run,
                          scan_data,
                          scan_kwargs,
                          scan_type='energy')
    

    def add_ANGLE_RC(self,
                     bs_run,
                     scan_data,
                     scan_kwargs):
        """Add data specific to ANGLE_RC scans"""

        if self._verbose:
            print('Adding ANGLE_RC...')

        self._add_step_rc(bs_run,
                          scan_data,
                          scan_kwargs,
                          scan_type='angle')

    
    def _add_step_rc(self,
                     bs_run,
                     scan_data,
                     scan_kwargs,
                     scan_type):

        if scan_type.lower() not in ['energy', 'angle']:
            raise ValueError('Only Energy and Angle step rocking curves accepted.')
        scan_type = scan_type.capitalize()

        # Load more useful metadata specific to XRF_FLY
        scan = bs_run.start['scan']

        # Generate scan table information
        table_labels = ['Scan Inputs', f'{scan_type} Step', f'{scan_type} Range', 'Detectors', 'Pixel Dwell']
        scan_inputs = scan['scan_input']
        input_str = (f"{np.round(scan_inputs[0], 0)}, {np.round(scan_inputs[1], 0)},"
                     + f"\n{int(scan_inputs[2])}, {np.round(scan_inputs[3], 2)}")
        all_dets = [det for det in scan['detectors']]
        useful_dets = [det for det in all_dets if det not in ['nano_vlm', 'ring_current']]
        
        if scan_type.lower() == 'energy':
            scan_inputs = [float(en) for en in scan_inputs]
            if scan_inputs[0] < 1e3:
                scan_inputs[0] *= 1e3
                scan_inputs[1] *= 1e3
            rocking = np.linspace(*scan_inputs[:2], int(scan_inputs[2]))
            rocking = np.round(rocking.astype(float), 1)     
            units = 'eV'
        elif scan_type.lower() == 'angle':
            scan_inputs = [float(an) for an in scan_inputs]
            CONVERTED = False
            if 'primary' in bs_run:
                if bs_run['primary']['config']['nano_stage_th']['nano_stage_th_motor_egu'][0] == 'mdeg': 
                    scan_inputs[0] /= 1e3
                    scan_inputs[1] /= 1e3
                    CONVERTED = True
            if not CONVERTED and np.any(np.abs(scan_inputs[:1]) > 200):
                scan_inputs[0] /= 1e3
                scan_inputs[1] /= 1e3  

            rocking = np.linspace(*scan_inputs[:2], int(scan_inputs[2]))
            rocking = np.round(rocking.astype(float), 4)                   
            units = 'deg'
        scan_step = np.mean(np.diff(rocking))
        scan_range = np.round(np.max(rocking) - np.min(rocking), 0)    

        table_values = [input_str,
                        f'{scan_step} {units}',
                        f'{scan_range} {units}',
                        ', '.join(useful_dets),
                        f"{scan['dwell']} sec"
                        ]
        
        # Generate images
        images, max_widths = [], []
        if scan_data['exit_status'] == 'success':
            if 'nano_vlm' in all_dets:
                fig = self._get_vlm_plot(bs_run,
                                         max_width=self._max_plot_width)
                if fig is not None:
                    images.append(self._figure_to_image(fig))
                    max_widths.append(self._max_plot_width)

            # Load AD Data
            for det in useful_dets:
                if (det not in ['dexela', 'merlin']
                    or 'primary' not in bs_run
                    or bs_run.stop['exit_status'] != 'success'):
                    continue

                for key in ['sclr_i0', 'sclr_im']:
                    if key in bs_run['primary']['data']:
                        sclr = bs_run['primary']['data'][key][:].astype(float)
                        break
                else:
                    sclr = np.ones(len(rocking), dtype=float)

                data, roi_str = self._load_and_process_ad_data(bs_run,
                                                            det,
                                                            sclr)

                # Plot_data
                if data is not None:
                    fig = self._get_line_plot(
                            [rocking],
                            [data.squeeze()],
                            title=f"Scan {bs_run.start['scan_id']}\n{roi_str}",
                            y_label='Normalized Intensity [a.u.]',
                            x_label=f'{scan_type} [{units}]',
                            max_height=self._max_height,
                            max_width=self._max_plot_width)
                    images.append(self._figure_to_image(fig))
                    max_widths.append(self._max_plot_width)   

        # Determine if a new page needs to be added
        self.request_cell_space(num_images=len(images))
        
        # Add base scan information
        self.add_BASE_SCAN(scan_data, scan_kwargs)

        # Draw table
        reset_y = self.y
        self.set_xy(self.x + self._gap_width, self.y)
        self.set_font(style='B', size=self._table_header_font_size)
        self.cell(h=self._table_header_height, new_x='LEFT', new_y='NEXT', text=f"Scan Data")
        self.set_font(size=self._table_font_size)
        l_margin = self.l_margin 
        self.l_margin = self.x # Temp move to set table location. Probably bad
        with self.table(
                first_row_as_headings=False,
                line_height=self._table_cell_height,
                col_widths=self._scan_table_cols_rc,
                width=self._scan_table_width,
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

        # Draw images
        self.draw_images(images, max_widths, self._max_height)
    

    def add_STATIC_XRD(self,
                       bs_run,
                       scan_data,
                       scan_kwargs):
        """Add data specific to STATIC_XRF scans"""

        if self._verbose:
            print('Adding STATIC_XRD...')

        # Load more useful metadata specific to XRF_FLY
        scan = bs_run.start['scan']

        # Generate scan table information
        table_labels = ['Scan Inputs', 'Image Num', 'Image Dwell', 'Detectors']
        scan_inputs = scan['scan_input']
        input_str = f"{scan_inputs[0]}, {scan_inputs[1]}"
        all_dets = [det for det in scan['detectors']]
        useful_dets = [det for det in all_dets if det not in ['nano_vlm', 'ring_current']]
        roi_dets = [det for det in useful_dets if det in ['merlin', 'dexela']]
        table_values = [input_str,
                        f"{scan_inputs[0]}",
                        f"{scan['dwell']} sec",
                        ', '.join(useful_dets),
                        ]
        
        # Generate images
        images, max_widths = [], []
        if scan_data['exit_status'] == 'success':
            if 'nano_vlm' in all_dets:
                fig = self._get_vlm_plot(bs_run,
                                        max_width=self._max_plot_width)
                if fig is not None:
                    images.append(self._figure_to_image(fig))
                    max_widths.append(self._max_plot_width)

            # Load AD Data
            for det in roi_dets:
                data, roi_str = self._load_and_process_ad_data(bs_run,
                                                            det,
                                                            None,
                                                            max_axes=(0, 1))
                
                # Check data
                if data is None:
                    continue

                clims = [np.mean(data) - 2 * np.std(data), np.mean(data) + 2 * np.std(data)]
                if clims[0] < 0:
                    clims[0] = 0

                # Plot_data
                figsize = [self._max_plot_width / 15, self._max_height / 15]
                fig, ax = plt.subplots(figsize=figsize, layout='tight', dpi=200)
                fontsize = 12

                im = ax.imshow(data, vmin=clims[0], vmax=clims[1])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=0.1, pad=0.1)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=fontsize)
                ax.set_aspect('equal')
                ax.set_aspect('equal')
                ax.tick_params(axis='x', labelbottom=False)
                ax.tick_params(axis='y', labelleft=False)  
                ax.set_xlabel("image x", fontsize=fontsize)
                ax.set_ylabel("image y", fontsize=fontsize)
                ax.set_title(f"Scan {bs_run.start['scan_id']}\n{roi_str}", fontsize=fontsize, pad=5)

                images.append(self._figure_to_image(fig))
                max_widths.append(self._max_plot_width)

        # Determine if a new page needs to be added
        self.request_cell_space(num_images=len(images))
        
        # Add base scan information
        self.add_BASE_SCAN(scan_data, scan_kwargs)

        # Draw table
        reset_y = self.y
        self.set_xy(self.x + self._gap_width, self.y)
        self.set_font(style='B', size=self._table_header_font_size)
        self.cell(h=self._table_header_height, new_x='LEFT', new_y='NEXT', text=f"Scan Data")
        self.set_font(size=self._table_font_size)
        l_margin = self.l_margin 
        self.l_margin = self.x # Temp move to set table location. Probably bad
        with self.table(
                first_row_as_headings=False,
                line_height=self._table_cell_height,
                col_widths=self._scan_table_cols_rc,
                width=self._scan_table_width,
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

        # Draw images
        self.draw_images(images, max_widths, self._max_height)
    

    ### Standardized Plotting ###
    
    def _get_vlm_plot(self,
                      bs_run,
                      max_height=None,
                      max_width=None,
                      scale=0.345 # um / px
                      ):
        
        if max_height is None:
            max_height = self._max_height
        if max_width is None:
            max_width = self._max_width
        
        # Check for data
        if ('camera_snapshot' not in bs_run
            or 'nano_vlm_image' not in bs_run['camera_snapshot']['data']):
            warn_str = ("WARNING: VLM image data missing for scan "
                        + f"{bs_run.start['scan_id']}. "
                        + "Proceeding without changes.")
            print(warn_str)
            return None

        # Get the data
        data = bs_run['camera_snapshot']['data']['nano_vlm_image'][:].squeeze()
        if data.ndim > 2:
            data = data[0]
        marker = tuple([bs_run.start['scan']['detectors']['nano_vlm'][f'cross_position_{a}'] for a in ['x', 'y']])
        title=f"Scan {bs_run.start['scan_id']}\nVLM Snapshot"

        figsize = [max_width / 15, max_height / 15]
        fig, ax = plt.subplots(figsize=figsize, layout='tight', dpi=200)
        fontsize = 12

        # Add image
        # clims = (np.min(data), np.min([np.max(data), 1e4]))
        clims = [np.mean(data) - 2 * np.std(data), np.mean(data) + 2 * np.std(data)]
        if clims[0] < 0:
            clims[0] = 0

        im = ax.imshow(data, cmap='gray', vmin=clims[0], vmax=clims[1])
    
        # Add marker
        if not all([m is None for m in marker]):
            ax.scatter(*marker, marker='+', lw=2, s=100, c='r')

        # Format scalbar
        scalebar = ScaleBar(0.345, # Example: 1 pixel = 0.2 microns
                            'um',
                            fixed_value=100,
                            color='r',
                            box_alpha=0,
                            width_fraction=0.02,
                            font_properties={'size' : fontsize},
                            location='lower right') 
        ax.add_artist(scalebar)
        
        # Format axes
        ax.set_aspect('equal')
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelleft=False)  
        ax.set_xlabel("x-axis", fontsize=fontsize)
        ax.set_ylabel("y-axis", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize, pad=5)
                    
        return fig

    
    def _get_xrf_map_plot(self,
                          bs_run,
                          scan_args,
                          roi,
                          roi_label,
                          transposed,
                          sclr,
                          sclr_key,
                          full_motors,
                          motor_units,
                          energy=None,
                          **kwargs):

        # Load data around ROI
        if bs_run.start['scan']['type'].split('_')[-1].lower() == 'fly':
            data = np.sum(bs_run['stream0']['data']['xs_fluor'][..., :7, roi], axis=(-2, -1,), dtype=np.float32)
        else:
            data = self._load_and_reshape_step_data(bs_run, roi)

        data /= sclr
        clims = [np.min(data), np.max(data)]
        en_int = energy[roi]
        roi_str = f"{roi_label}: {int(en_int[0])} - {int(en_int[-1])} eV"

        fig = self._get_mapped_plot(bs_run,
                                    scan_args,
                                    data,
                                    transposed,
                                    title=f"Scan {bs_run.start['scan_id']}\n{roi_str}",
                                    int_label='Normalized Intensity [a.u.]',
                                    y_label=f"{full_motors[1]} [{motor_units[1]}]",
                                    x_label=f"{full_motors[0]} [{motor_units[0]}]",
                                    **kwargs)

        return fig

    
    def _get_sclr_map_plot(self,
                           bs_run,
                           scan_args,
                           roi,
                           roi_label,
                           transposed,
                           sclr,
                           sclr_key,
                           full_motors,
                           motor_units,
                           **kwargs):

        if bs_run.start['scan']['type'].split('_')[-1].lower() == 'fly':
            data = bs_run['stream0']['data'][roi][:].astype(np.float32)
        else:
            data = self._load_and_reshape_step_data(bs_run, roi)
        if roi != sclr_key: # Normalize if not plotting the normalization
            data /= sclr
            int_label = 'Normalized Intensity [a.u.]'
        else:
            int_label = 'Intensity [a.u.]'
        roi_str = f"Scaler: {roi_labels[roi_ind]}"

        fig = self._get_mapped_plot(bs_run,
                                    scan_args,
                                    data,
                                    transposed,
                                    title=f"Scan {bs_run.start['scan_id']}\n{roi_str}",
                                    int_label=int_label,
                                    y_label=f"{full_motors[1]} [{motor_units[1]}]",
                                    x_label=f"{full_motors[0]} [{motor_units[0]}]",
                                    **kwargs)

        return fig


    def _get_ad_map_plot(self,
                         bs_run,
                         scan_args,
                         roi,
                         roi_label,
                         transposed,
                         sclr,
                         sclr_key,
                         full_motors,
                         motor_units,
                         **kwargs):
        
        # Load Data
        if bs_run.start['scan']['type'].split('_')[-1].lower() == 'fly':
            data, roi_str = self._load_and_process_ad_data(bs_run,
                                                            roi,
                                                            sclr)
        else:
            data = self._load_and_reshape_step_data(bs_run, roi)
            roi_str = f'{roi}_max'

        if data is None:
            return None

        # Get mapped plot
        fig = self._get_mapped_plot(bs_run,
                                    scan_args,
                                    data,
                                    transposed,
                                    title=f"Scan {bs_run.start['scan_id']}\n{roi_str}",
                                    int_label='Normalized Intensity [a.u.]',
                                    y_label=f"{full_motors[1]} [{motor_units[1]}]",
                                    x_label=f"{full_motors[0]} [{motor_units[0]}]",
                                    **kwargs)

        return fig
    

    def _get_mapped_plot(self,
                         bs_run,
                         scan_args,
                         data,
                         transposed,
                         colornorm='linear',
                         title=None,
                         int_label=None,
                         y_label=None,
                         x_label=None,
                         max_height=None,
                         max_width=None,
                         ):

        if max_height is None:
            max_height = self._max_height
        if max_width is None:
            max_width = self._max_plot_width

        # Parsing data shapes
        start_shape = tuple([int(s) for s in bs_run.start['scan']['shape']])[::-1]
        # 1D data is fine
        if 1 in data.shape:
            pass
        else:
            # Fly in y, transponsed data
            if transposed:
                data = data.T
                start_shape = start_shape[::-1]
                # Is the data complete?
                if start_shape != data.shape:
                    full_data = np.zeros(start_shape)
                    full_data[:] = np.nan
                    for i in range(data.shape[1]):
                        full_data[:, i] = data[:, i]
                    data = full_data
            else: # Fly in x
                # Is the data complete?
                if start_shape != data.shape:
                    full_data = np.zeros(start_shape)
                    full_data[:] = np.nan
                    for i in range(data.shape[0]):
                        full_data[i] = data[i]
                    data = full_data
        
        # Generate plot
        figsize = [max_width / 15, max_height / 15]
        fig, ax = plt.subplots(figsize=figsize, layout='tight', dpi=200)
        fontsize = 12

        exts = [scan_args[1] - scan_args[0], scan_args[4] - scan_args[3]]
        if bs_run.start['scan']['type'].split('_')[-1].lower() == 'fly':
            nums = [scan_args[2], scan_args[5]]
            steps = [ext / (num - 1) if num != 1 else 0 for ext, num in zip(exts, nums)]
        else:
            steps = [scan_args[2], scan_args[5]]
            nums = [int(e / s) + 1 for (e, s) in zip(exts, steps)]

        if 1 in data.shape: # Plot plot
            if nums[0] != 1:
                ax.plot(np.linspace(*scan_args[:2], int(nums[0])), data.squeeze())
            elif nums[1] != 1:
                ax.plot(np.linspace(*scan_args[3:5], int(nums[1])), data.squeeze())
            else:
                ax.scatter(scan_args[0], data.squeeze()) # Not a very good map...
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            ax.set_ylabel(int_label, fontsize=fontsize)
        else: # Plot image
            extent = [scan_args[0] - steps[0] / 2,
                      scan_args[1] + steps[0] / 2,
                      scan_args[4] + steps[1] / 2,
                      scan_args[3] - steps[1] / 2]
            clims = [np.min(data), np.max(data)]

            # Change colorscale
            if colornorm == 'log':
                log_min = 1 / np.max(sclr)
                if 'dexela' in title:
                    log_min *= 1e2
                data[data <= 0] = log_min
                clims[0] = log_min

            im = ax.imshow(data, extent=extent, norm=colornorm, vmin=clims[0], vmax=clims[1])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.1, pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            if colornorm == 'linear':
                cbar.formatter.set_powerlimits((-2, 2))
            cbar.ax.tick_params(labelsize=fontsize)
            ax.set_aspect('equal')
            ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize, pad=15)
        
        return fig


    def _get_line_plot(self,
                       x, # array([[x0, x1, ...]])
                       y, # array([[y0, y1, ...]])
                       marker=None, # array([[x0, y0], [x1, y1], ...])
                       title=None,
                       labels=None, # [label0, label1, ...]
                       y_label=None,
                       x_label=None,
                       max_height=None,
                       max_width=None,
                       ):
        
        if max_height is None:
            max_height = self._max_height
        if max_width is None:
            max_width = self._max_plot_width
        
        if labels is None:
            labels = [None,] * len(x)
        
        fig, ax = plt.subplots(figsize=(max_width / 15, max_height / 15), tight_layout=True, dpi=200)
        fontsize=12           
        ax.plot(np.asarray(x).T, np.asarray(y).T, label=labels)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_title(title, fontsize=fontsize, pad=5)

        if marker is not None:
            ax.scatter(*np.asarray(marker).T, marker='*', s=50, c='r')
        if len(labels) > 1:
            ax.legend()

        return fig
    

    def _load_and_reshape_step_data(self,
                                    bs_run,
                                    roi):
        # Probably overkill ensuring the pixel grid is correct
        data = np.zeros(bs_run.start['scan']['shape'][::-1]) # must be reversed for [y, x]
        data[:] = np.nan # For unfinished plotting purposes

        if isinstance(roi, slice):
            flat_data = np.asarray([bs_run['primary']['data'][f'xs_channel0{ind + 1}_fluor'][..., roi] for ind in range(7)],
                                    dtype=np.float32).sum(axis=(0, -1))
        elif 'image' in roi:
            # Untested as of 2025-08-27 due to handler errors 
            flat_data = np.asarray(bs_run['primary']['data'][roi], dtype=float).squeeze().max(axis=(-2, -1))
        else:
            flat_data = np.asarray(bs_run['primary']['data'][roi], dtype=float)
        flat_x = bs_run['primary']['data'][bs_run.start['scan']['fast_axis']['motor_name']][:]
        flat_y = bs_run['primary']['data'][bs_run.start['scan']['slow_axis']['motor_name']][:]
        grid_x = np.linspace(*bs_run.start['scan']['scan_input'][0:2], data.shape[1])
        grid_y = np.linspace(*bs_run.start['scan']['scan_input'][3:5], data.shape[0])
        
        for x, y, d in zip(flat_x, flat_y, flat_data):
            ind_x = np.argmin(np.abs(x - grid_x))
            ind_y = np.argmin(np.abs(y - grid_y))
            data[ind_y, ind_x] = d
        
        return data
    
    def _load_and_process_ad_data(self,
                                  bs_run,
                                  roi,
                                  sclr,
                                  max_axes=(-2, -1)
                                  ):

        scan_type = bs_run.start['scan']['type']
        if scan_type == 'XRF_FLY':
            stream_name = 'stream0'
        else:
            stream_name = 'primary'                          

        # Check for data
        if stream_name not in bs_run:
            warn_str = (f"WARNING: Stream {stream_name} missing for "
                        + f"scan {bs_run.start['scan_id']}. "
                        + "Proceeding without changes.")
            print(warn_str)
            return None, ''
        elif f'{roi}_image' not in bs_run[stream_name]['data']:
            warn_str = (f"WARNING: Key not in '{stream_name}' for "
                        + f"{roi} data from scan "
                        + f"{bs_run.start['scan_id']}. Proceeding "
                        + "without changes.")
            print(warn_str)
            return None, ''

        # Load dark-field if there
        if 'dark' in bs_run:
            # Assume that this will always work for now...
            try:
                dark = bs_run['dark']['data'][f'{roi}_image'][:].astype(np.float32)
                dark = np.median(dark.reshape((-1, *dark.shape[-2:])), axis=0)
            except Exception as e:
                warn_str = ('WARNING: Error loading dark-field data '
                            + f'from {roi} in tiled. Proceeding '
                            + f'without dark-field. \n{e}')
                print(warn_str)
                dark = None
        else:
            dark = None
        
        scan_type = bs_run.start['scan']['type']     

        try:
            # Manual load data of fly-scanning map
            if scan_type == 'XRF_FLY':
                # Pseudo binning
                # Saves memory, but at the cost of time
                data_shape = bs_run[stream_name]['data'][f'{roi}_image'].shape
                data_slicing = tuple([slice(None), slice(None)]
                                    + [slice(None, None, int(s / 500) + 1)
                                       for s in data_shape[-2:]]) 

                data = manual_load_data(int(bs_run.start['scan_id']),
                                        data_keys=[f'{roi}_image'],
                                        repair_method='fill',
                                        verbose=False)[0][f'{roi}_image']
                data = np.asarray([d[data_slicing[1:]] for d in data]).astype(np.float32)
                if dark is not None:
                    dark = dark[data_slicing[-2:]]

            # Manual load data of step-scanning rocking curve
            elif scan_type in ['ANGLE_RC', 'ENERGY_RC']:
                data = load_step_rc_data(int(bs_run.start['scan_id']), verbose=False)[0]
                data = np.asarray(data[f'{roi}_image'], dtype=np.float32)
            
            # Don't know an hope tiled works
            else:
                data = bs_run[stream_name]['data'][f'{roi}_image'][:].astype(np.float32)

            # Process data
            if all([ax in [-2, -1, 3, 4] for ax in max_axes]):
                null_map = np.all(data == 0, axis=max_axes)
            else:
                null_map = np.zeros(data.shape[:2], dtype=np.bool_)
            if dark is None:
                dark = np.min(data[~null_map], axis=0) # approximate
            data -= dark
            data = median_filter(data, size=(1, 1, 2, 2)) # Denoise. Also takes a long time
            data = np.max(data, axis=max_axes)
            if sclr is not None:
                data /= sclr
            if data.shape == null_map.shape:
                data[null_map] = 0
            roi_str = f'{roi}_max'
        
        except Exception as e:
            err_str = (f"{e}: Error loading {roi} "
                        + f"from scan {bs_run.start['scan_id']}. "
                        + f"Proceeding without changes.")
            print(err_str)
            data = None
            roi_str = ''
        
        return data, roi_str


    ### Drawing Images from Figures ###

    def draw_images(self, images, max_widths, max_height):
        
        # Record current position
        reset_y = self.y
        self.set_xy(self.x + self._gap_width, self.y) # Add gap after table

        if len(images) == 0:
            image_ind = -1

        # Iterate through images       
        for image_ind, (image, width) in enumerate(zip(images, max_widths)):

            # if (image_ind - 1) % 3 in [1, 2]:
            #     width = np.min([width, self.epw / 3])

            # Determine image size which best fills available space
            width_at_max_height = (image.width / image.height) * max_height
            if width_at_max_height > width:
                image_height = width * (image.height / image.width)
                image_width = width
            else:
                image_height = max_height
                image_width = (image.width / image.height) * max_height

            # Plot with tables
            if image_ind == 0:
                if self._verbose:
                    print(f'Drawing image {image_ind} at first index.') 
                image_x = self.x + ((self.w - self.r_margin) - self.x - image_width) / 2
                self.image(image, x=image_x, h=image_height, keep_aspect_ratio=True)
                # Start new line of images
                self.set_xy(self.l_margin, reset_y + self._max_height + self._offset_height)
                reset_y = self.y

            # Plot left of three
            elif (image_ind - 1) % 3 == 0:
                if self._verbose:
                    print(f'Drawing image {image_ind} at on left.') 
                image_x = self.l_margin
                self.image(image, x=image_x, h=image_height, keep_aspect_ratio=True)
                self.set_xy(self.l_margin, reset_y)

            # Plot middle of three
            elif (image_ind - 1) % 3 == 1:
                if self._verbose:
                    print(f'Drawing image {image_ind} at at center.') 
                image_x = (self.w - image_width) / 2
                self.image(image, x=image_x, h=image_height, keep_aspect_ratio=True)
                self.set_xy(self.l_margin, reset_y)

            # Plot right of three    
            elif (image_ind - 1) % 3 == 2:
                if self._verbose:
                    print(f'Drawing image {image_ind} at on right.') 
                image_x = self.w - self.r_margin - image_width
                self.image(image, x=image_x, h=image_height, keep_aspect_ratio=True)
                # Start new line of images
                self.set_xy(self.l_margin, reset_y + self._max_height + self._offset_height)
                reset_y = self.y
        
        # Cleanup
        if image_ind == 0 or (image_ind - 1) % 3 == 2:
            self.set_xy(self.l_margin, self.y + 0.5)
            if self._verbose:
                print(f'First cleanup call after drawing first or last image.')
        else:
            self.set_xy(self.l_margin, self.y + self._max_height + self._gap_height)
            if self._verbose:
                print('Second cleanup call after drawing other images or none.') 
    

    def _figure_to_image(self, figure):

        # Converting Figure to an image:
        canvas = FigureCanvas(figure)
        canvas.draw()
        image = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        plt.close(figure)
        
        return image
    

    ### Extract information from tiled ###

    def get_proposal_scan_data(self,
                               scan_id):
        """Get proposal information for header."""

        start = c[int(scan_id)].start
        exp_md = {}

        # Add proposal information
        for key in ['proposal_id', 'title', 'pi_name']:
            if key in start['proposal']:
                exp_md[key] = start['proposal'][key]
            else:
                exp_md[key] = None
        # Add cycle
        if 'cycle' in start:
            exp_md['cycle'] = start['cycle']
        else:
            exp_md['cycle'] = None
        
        UPDATED_EXP_MD = False
        for key, value in exp_md.items():
            if key not in self.exp_md:
                UPDATED_EXP_MD = True
            elif self.exp_md[key] != value:
                UPDATED_EXP_MD = True
            self.exp_md[key] = value
            
        return UPDATED_EXP_MD

    def get_start_scan_data(self,
                            bs_run):
        """Get metadata from start document."""
        
        scan_meta_data = {}

        start = bs_run.start
        stop = bs_run.stop

        # For odd cases without start/stop docs
        if start is None:
            start = {}
        if stop is None:
            stop = {}

        if 'scan_id' in start:
            scan_meta_data['scan_id'] = start['scan_id']
        else:
            scan_meta_data['scan_id'] = 'ERROR'
        
        if 'time' in start:
            scan_meta_data['start_time'] = start['time']
        else:
            scan_meta_data['start_time'] = None

        if 'scan' in start:
            if 'type' in start['scan']:
                scan_meta_data['scan_type'] = start['scan']['type']
            else:
                scan_meta_data['scan_type'] = 'UNKNOWN'
        else:
            scan_meta_data['scan_type'] = 'UNKNOWN'

        if 'time' in stop:
            scan_meta_data['stop_time'] = stop['time']
        else:
            scan_meta_data['stop_time'] = None
        
        if 'exit_status' in stop:
            scan_meta_data['exit_status'] = stop['exit_status']
        else:
            scan_meta_data['exit_status'] = 'unknown'

        return scan_meta_data
    

    def get_baseline_scan_data(self,
                               bs_run):
        """Get reference data from baseline"""
        scan_base_data = {}

        if 'baseline' in bs_run:
            baseline = bs_run['baseline']
            NO_BASELINE = False
        else:
            warn_str = ("WARNING: Baseline not found in scan "
                        + f"{bs_run.start['scan_id']}. Reference data "
                        + "could not be determined.")
            print(warn_str)
            # Create empty simulacrum
            baseline = {'data' : {},
                        'config' : {'nano_stage' : {}}}
            NO_BASELINE = True

        # Stage values!
        for key in ['x', 'y', 'z', 'topx', 'topz', 'th', 'sx', 'sy', 'sz']:
            full_key = f'nano_stage_{key}'
            if f'{full_key}_user_setpoint' in baseline['data']:
                scan_base_data[key] = baseline['data'][f'{full_key}_user_setpoint'][0]
            else:
                scan_base_data[key] = None
            if f'{full_key}_motor_egu' in baseline['config']['nano_stage']:
                unit = baseline['config']['nano_stage'][f'{full_key}_motor_egu'][0]
                scan_base_data[f'{key}_units'] = unit   
            else:
                scan_base_data[f'{key}_units'] = None
        
        # Try to load theta from start if not in baseline
        if (scan_base_data['th'] is None
            and 'scan' in bs_run.start
            and 'theta' in bs_run.start['scan']):
            th = bs_run.start['scan']['theta']['val']
            units = bs_run.start['scan']['theta']['units']
            scan_base_data['th'] = th
            scan_base_data['th_units'] = units

        # Energy values!
        if 'energy_energy_setpoint' in baseline['data']:
            en = baseline['data']['energy_energy_setpoint'][0]
            if en < 1e3:
                en *= 1e3
            scan_base_data['energy'] = en
            scan_base_data['energy_units'] = 'eV'
        elif 'scan' in bs_run.start and 'energy' in bs_run.start['scan']:
            en = bs_run.start['scan']['energy']
            if isinstance(en, list):
                en = en[0]
            if en < 1e3:
                en *= 1e3
            scan_base_data['energy'] = en
            scan_base_data['energy_units'] = 'eV'
        else:
            scan_base_data['energy'] = None
            scan_base_data['energy_units'] = None

        # Check th units:
        if scan_base_data['th_units'] == 'mdeg':
            scan_base_data['th'] = np.round(scan_base_data['th'] / 1e3, 3)
            scan_base_data['th_units'] = 'deg'
        
        # SDD position value. sdd_dist will be used if available
        if 'nano_det_x' in baseline['data']:
            scan_base_data['sdd_x'] = baseline['data']['nano_det_x'][0]
            scan_base_data['sdd_dist'] = baseline['data']['nano_det_sample2detector'][0]
            scan_base_data['sdd_x_units'] = baseline['config']['nano_det']['nano_det_x_motor_egu'][0]
        else:
            scan_base_data['sdd_x'] = None
            scan_base_data['sdd_dist'] = None
            scan_base_data['sdd_x_units'] = None

        # Get attenuators. Only Al and Si are supported
        attenuators = ['Al_050um', 'Al_100um', 'Al_250um', 'Al_500um', 'Si_250um', 'Si_650um']
        attenuators = [f'attenuators_{att}' for att in attenuators]
        for att in attenuators:
            if att in baseline['data']:
                scan_base_data[att] = baseline['data'][att][0]
            else:
                scan_base_data[att] = None

        return scan_base_data


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
                fly scan plotting. Can include 'merlin' or 'dexela'
                and by default will include neither.
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
    get_kwargs = lambda func : func.__code__.co_varnames[len(func.__defaults__) - 1 : func.__code__.co_argcount]

    # Parse kwargs to make sure they are useful. Add functions and methods as necessary
    useful_kwargs = (list(get_kwargs(SRXScanPDF.add_scan))
                     + list(get_kwargs(SRXScanPDF._add_xrf_general))
                     + list(get_kwargs(_find_xrf_rois)))
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
            start_proposal_id = c[start_id].start['proposal']['proposal_id']

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
                print(f'Current scan ID {current_id} not yet finished. Waiting 5 minutes...')
                ttime.sleep(300)
                continue
            
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
            scan_report.add_scan(current_id, **kwargs)
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
