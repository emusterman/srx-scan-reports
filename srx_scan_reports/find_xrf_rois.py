
import numpy as np
from scipy.signal import find_peaks
import skbeam.core.constants.xrf as xrfC
from itertools import combinations_with_replacement
from scipy.stats import mode

from . import c

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
def find_xrf_rois(xrf,
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
                line = None
                if '_' in el:
                    el, line = el.split('_')
                el = el.capitalize()
                if el in possible_elements:
                    if line is not None:
                        found_elements.append(xrfC.XrfElement(el))
                        specific_lines.append('_'.join([el, line]))
                        num_interesting_rois += 1
                    else:
                        if xrfC.XrfElement(el) not in found_elements:
                            found_elements.append(xrfC.XrfElement(el))
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