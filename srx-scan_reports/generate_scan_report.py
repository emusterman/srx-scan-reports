import numpy as np
import time as ttime
from tqdm import tqdm

from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from scipy.signal import find_peaks
import skbeam.core.constants.xrf as xrfC


from tiled.client import from_profile

# Access data
c = from_profile('srx')

# Get elemental information
interestinglist = ['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                   'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                   'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                   'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                   'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                   'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                   'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                   'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']

elements = [xrfC.XrfElement(el) for el in interestinglist]
edges = ['k', 'l1', 'l2', 'l3']
lines = ['ka1', 'kb1', 'la1', 'lb1', 'lb3', 'lb2', 'lg1', 'lb4', 'ma1']
major_lines = ['ka1', 'la1', 'ma1']

all_edges, all_edges_names, all_lines, all_lines_names = [], [], [], []
for el in elements:
    for edge in edges:
        edge_en = el.bind_energy[edge] * 1e3
        if 4e3 < edge_en < 22e3:
            all_edges.append(edge_en)
            all_edges_names.append(f'{el.sym}_{edge.capitalize()}')

# Based on line value for pseudo intensity!
for line in lines:
    for el in elements:
        line_en = el.emission_line[line] * 1e3
        if 1e3 < line_en < 20e3: 
            all_lines.append(line_en)
            all_lines_names.append(f'{el.sym}_{line}')


class SRXScanPDF(FPDF):

    def __init__(self):
         FPDF.__init__(self)

         self.exp_md = {}

    def header(self):
        self.set_font('helvetica', size=12)
        

        self.cell(w=0, new_x='LMARGIN', new_y='NEXT', text=f"Proposal # : {self.exp_md['proposal_id']}", align='R')
        self.cell(w=0, new_x='LMARGIN', new_y='NEXT', text=f"Proposal Title : {self.exp_md['title']}", align='R')
        self.cell(w=0, new_x='LMARGIN', new_y='NEXT', text=f"Proposal PI : {self.exp_md['pi_name']}", align='R')

        self.image('srx_logo.png', x=self.l_margin, y=5, h=17.5, keep_aspect_ratio=True)

        self.set_line_width(0.5)
        self.set_draw_color(0, 0, 0)
        self.line(x1=self.l_margin, y1=self.y + 2.5, x2=self.epw + self.l_margin, y2=self.y + 2.5)

    def footer(self):
        self.set_xy(self.x, -15)
        self.set_font('helvetica', size=10)
        self.cell(0, 10, f'Page{self.page_no()}/{{nb}}', align='C')
        self.cell(0, 10, f'Generated on {ttime.ctime()}', align='R')        

    
    def add_scan(self,
                 scan_id,
                 include_peakups=True,
                 include_unknowns=True,
                 include_failures=True):
        """Add a scan from a scan_id"""

        bs_run = c[int(scan_id)]

        # Extract metadata from start
        scan_data = self._get_start_scan_data(bs_run)
        # Extract reference data from baseline
        scan_data.update(self._get_baseline_scan_data(bs_run))
        
        # Skip failed scans if not including failures
        if (not include_failures
            and scan_data['exit_status'] != 'success'):
            return
        
        # Check proposal data and update header as needed.
        self.get_proposal_scan_data(bs_run)

        # Build report entry

        if scan_data['scan_type'] == 'XRF_FLY':
            # Check for room. Value is empirically measured...
            if self.h - self.y - self.b_margin < 80:
                self.add_page()
            start_y = self.y
            self.add_BASE_SCAN(scan_data)
            self.add_XRF_FLY(bs_run, scan_data)
            end_y = self.y
            # print(f'Cell height is {end_y - start_y}')
        elif scan_data['scan_type'] == 'XAS_STEP':
            if self.h - self.y - self.b_margin < 80:
                self.add_page()
            self.add_BASE_SCAN(scan_data)
            self.add_XAS_STEP(bs_run, scan_data)
        elif scan_data['scan_type'] == 'XRF_STEP':
            self.add_BASE_SCAN(scan_data)
            self.add_XRF_STEP(bs_run, scan_data)
        elif scan_data['scan_type'] == 'ANGLE_RC':
            self.add_BASE_SCAN(scan_data)
            self.add_ANGLE_RC(bs_run, scan_data)
        elif scan_data['scan_type'] == 'ENERGY_RC':
            self.add_BASE_SCAN(scan_data)
            self.add_ENERGY_RC(bs_run, scan_data)
        elif scan_data['scan_type'] == 'XAS_FLY':
            self.add_BASE_SCAN(scan_data)
            self.add_XAS_FLY(bs_run, scan_data)
        elif scan_data['scan_type'] == 'PEAKUP' and include_peakups:
            if self.h - self.y - self.b_margin < 15:
                self.add_page()
            start_y = self.y
            self.add_BASE_SCAN(scan_data)
            end_y = self.y
            print(f'Cell height is {end_y - start_y}')
        elif include_unknowns:
            warn_str = (f"WARNING: Scan {scan_id} of type {scan_data['scan_type']} "
                        + "not yet support for SRX scan reports.")
            print(warn_str)
            self.add_BASE_SCAN(scan_data)
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

        for scan_id in tqdm(range(int(start_id), int(end_id) + 1)):
            self.add_scan(scan_id, **kwargs)

    
    def add_BASE_SCAN(self,
                      scan_data):
        """Add basic data for all scans"""

        # Generate starting line
        self.set_line_width(0.5)
        self.set_draw_color(0, 0, 0)
        self.set_xy(self.x, self.y + 2.5)
        self.line(x1=self.l_margin, y1=self.y, x2=self.epw + self.l_margin, y2=self.y)

        scan_labels = ['ID', 'Type', 'Status', 'Start', 'Stop', 'Duration']
        scan_values = [str(scan_data['scan_id']),
                       scan_data['scan_type'],
                       scan_data['exit_status'].upper(),
                       ttime.strftime('%b %d %H:%M:%S', ttime.localtime(scan_data['start_time'])),
                       ttime.strftime('%b %d %H:%M:%S', ttime.localtime(scan_data['stop_time'])),
                       ttime.strftime('%H:%M:%S', ttime.gmtime(scan_data['stop_time'] - scan_data['start_time']))]
        
        self.set_xy(self.x, self.y + 2.5)
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
                    self.set_font(style='BU', size=10)
                row.cell(f'{scan_labels[i]}: ')
                if i == 0:
                    self.set_font(style='U', size=10)
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
        
        self.set_xy(self.x, self.y + 2.5)
        reset_y = self.y
        self.set_font(style='B', size=10)
        self.cell(h=5, new_x='LMARGIN', new_y='NEXT', text=f"Reference Data")
        self.set_font(size=10)
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=5,
                col_widths=(22.5, 22.5),
                width=42.5,
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
        print(f'Base Table height is is {reset_y - self.y}')
        self.set_xy(self.x + table_width, reset_y)

    
    def add_TEST_SCAN(self,
                      bs_run):
        """Generalized testing scan"""
        pass

    
    def add_XRF_FLY(self,
                    bs_run,
                    scan_data,
                    data_slice=None):
        """Add data specific to XRF_FLY scans"""
        

        
        # Load more useful metadata specific to XRF_FLY
        scan = bs_run.start['scan']

        table_labels = ['Scan Input', 'Motors', 'Detectors', 'Pixel Dwell', 'Map Shape', 'Map Size', 'Step Sizes']
        args = scan['scan_input']
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

        reset_y = self.y
        self.set_xy(self.x + 2.5, self.y)
        self.set_font(style='B', size=10)
        self.cell(h=5, new_x='LEFT', new_y='NEXT', text=f"Scan Data")
        self.set_font(size=10)
        l_margin = self.l_margin 
        self.l_margin = self.x # Temp move to set table location. Probably bad
        # print(f'Before table {self.x=}, {self.y=}')
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=5,
                col_widths=(22.5, 37.5),
                width=55,
                align='L',
                text_align='LEFT'
                ) as table:
            for i in range(len(table_labels)):
                row = table.row()
                row.cell(table_labels[i])
                row.cell(table_values[i])
            table_width = table._width
        max_height = self.y - reset_y + 15 # Extra row and cell height
        print(f'{max_height=}')
        self.set_xy(self.x + table_width, reset_y)
        # print(f'After table {self.x=}, {self.y=}')
        self.l_margin = l_margin # Reset left margin
        max_width = self.w - self.r_margin - self.x

        # Do not even try any processing of failed scans. Too many potential failure points
        if (scan_data['exit_status'] == 'success'
            and 'stream0' in bs_run):
            
            # Auto ROI. May remove!!!
            if data_slice is None:
                if scan_data['energy'] is not None:
                    end = scan_data['energy'] / 10
                else:
                    end = 2048 # Limit of silicon
                
                cond_data = np.sum(bs_run['stream0']['data']['xs_fluor'][..., 200:int(0.85 * end)], axis=(0, 1, 2)).astype(np.float32)
                cond_data[cond_data < 1e-3] = 1e-3
                en = np.arange(200, int(0.85 * end) + 1) * 10
                peaks, proms = find_peaks(np.log(cond_data), prominence=1)
                peak_ints = [np.sum(cond_data[peak - 10 : peak + 10]) for peak in peaks] # blind pseudo-windowed
                sorted_peaks = [x for _, x in sorted(zip(peak_ints, peaks), key=lambda pair: pair[0], reverse=True)]
                # peak_en = en_ind[peaks[np.argmax(proms['prominences'])]]
                # peak_en = en_ind[peaks[np.argmax(cond_data[peaks])]]

                # Find elements
                en_tol = 50 # in eV
                found_elements = []
                for peak in sorted_peaks:
                    peak_en = en[peak]
                    # print(peak_en)

                    PEAK_FOUND = False
                    # Check if peak can be explained by already found elements
                    for el in found_elements:
                        for line in lines:
                            line_en = el.emission_line[line] * 1e3
                            if np.abs(peak_en - line_en):
                                PEAK_FOUND = True
                                break
                        else:
                            continue
                        break

                    # Otherwise check other elements based on strongest fluorescence
                    if not PEAK_FOUND:
                        for line in lines:
                            for el in elements:
                                line_en = el.emission_line[line] * 1e3
                                if np.abs(peak_en - line_en) < en_tol:
                                    PEAK_FOUND = True
                                    found_elements.append(el)
                                    break
                            else:
                                continue
                            break
                        else: # Very unlikely
                            found_elements.append(peak_en)

                # Generate new ROIS
                rois, rois_labels = [], []
                for el in found_elements:
                    if isinstance(el, int):
                        rois.append(slice(int(peak_en / 10) - 10, int(peak_en / 10) + 10))
                        rois_labels.append('Unknown')
                        continue

                    for line in lines:
                        line_en = el.emission_line[line] * 1e3
                        if 1e3 < line_en < scan_data['energy']:
                            rois.append(slice(int(line_en / 10) - 10, int(line_en / 10) + 10))
                            rois_labels.append(f'{el.sym}_{line}')

                # Singular hack
                data_slice = rois[0]
                data_label = rois_labels[0]
                en_int = (data_slice.start * 10, data_slice.stop * 10)



                
                # line_ind = np.argmin(np.abs(np.array(all_lines) - (peak_en * 10)))
                # el_line = all_lines_names[line_ind]
                # line_en = all_lines[line_ind]
                # #data_slice = slice(peak_en - 10, peak_en + 10)
                # data_slice = slice(int(line_en / 10) - 10, int(line_en / 10) + 10)

            # Load data around ROI
            data = np.sum(bs_run['stream0']['data']['xs_fluor'][..., data_slice], axis=(-2, -1,), dtype=np.float32)
            data /= bs_run['stream0']['data']['i0'][:].astype(np.float32)
            en_int = np.asarray(range(4096)[data_slice]) * 10

            # Check if scan completed
            if scan['shape'] != data.shape:
                full_data = np.zeros(tuple(scan['shape']))
                full_data[:] = np.nan
                for i in range(data.shape[0]):
                    full_data[i] = data[i]
                data = full_data

            # Plot data
            fig, ax = plt.subplots(figsize=(max_width / 15, max_height / 15), tight_layout=True)
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
                fig.colorbar(im, ax=ax)
                ax.set_aspect('equal')
                ax.set_ylabel(f"{full_motors[1]} [{motor_units[1]}]")
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.set_xlabel(f"{full_motors[0]} [{motor_units[0]}]", fontsize=fontsize)    
            # ax.set_title(f"Scan {bs_run.start['scan_id']}\nROI: {int(en_int[0])} - {int(en_int[-1])} eV", fontsize=fontsize)
            # ax.set_title(f"Scan {bs_run.start['scan_id']}\n{el_line}: {int(en_int[0])} - {int(en_int[-1])} eV", fontsize=fontsize)
            ax.set_title(f"Scan {bs_run.start['scan_id']}\n{data_label}: {int(en_int[0])} - {int(en_int[-1])} eV", fontsize=fontsize)

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
        self.set_xy(self.l_margin, reset_y + max_height + 2.5)


    def add_XAS_STEP(self,
                     bs_run,
                     scan_data):
        """Add data specific to XAS_STEP scans"""
        
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
        self.set_xy(self.x + 2.5, self.y)
        self.set_font(style='B', size=10)
        self.cell(h=5, new_x='LEFT', new_y='NEXT', text=f"Scan Data")
        self.set_font(size=10)
        l_margin = self.l_margin 
        self.l_margin = self.x # Temp move to set table location. Probably bad
        # print(f'Before table {self.x=}, {self.y=}')
        with self.table(
                # borders_layout="NONE",
                first_row_as_headings=False,
                line_height=5,
                col_widths=(22.5, 27.5),
                width=55,
                align='L',
                text_align='LEFT'
                ) as table:
            for i in range(len(table_labels)):
                row = table.row()
                row.cell(table_labels[i])
                row.cell(table_values[i])
            table_width = table._width
        max_height = self.y - reset_y + 15 # Extra row and cell height
        self.set_xy(self.x + table_width, reset_y)
        # print(f'After table {self.x=}, {self.y=}')
        self.l_margin = l_margin # Reset left margin
        max_width = self.w - self.r_margin - self.x

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
        fig, ax = plt.subplots(figsize=(max_width / 15, max_height / 15), tight_layout=True)
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
        self.set_xy(self.l_margin, reset_y + max_height + 2.5)
    
    def add_UNKOWN(self,
                   bs_run):
        pass


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
                               bs_run):
        """Get proposal information for header."""

        # if (not hasattr(self, 'exp_md')
        #     and not isinstance(self.exp_md, dict)):
        #     self.exp_md = {} 

        start = bs_run.start
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

scan_report.get_proposal_scan_data(c[164510])
scan_report.add_page()
scan_report.add_scan_range(164505, 164507)
# scan_report.add_scan(164524)

scan_report.output('test.pdf')