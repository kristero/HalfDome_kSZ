"""Aggregate measured comparisons without weighting them as a cosmic census.

Each reported median treats supported mass bins equally. The fixed halo holdout
is an empirical check within one simulation, not independent cosmic variance.
"""
import json
import csv
from pathlib import Path
import numpy as np
from fit_study import accepted, plain, SELECTIONS, FAMILIES
from projection_model import project_map, map_metrics
import measure as M
import pilot_physics as P

HERE = Path(__file__).resolve().parent


def main():
    summary_path = HERE/'products/study_summary.json'
    summary = json.loads(summary_path.read_text())
    cases = summary['bins']
    # These additional masks use existing coefficients and compact stack maps;
    # they require no refitting, resampling or new snapshot reads.
    maps = np.load(HERE/'projection_stacks.npz')
    coordinate = -M.EXTENT+(np.arange(M.NPIX)+.5)*2*M.EXTENT/M.NPIX
    xx, yy = np.meshgrid(coordinate, coordinate, indexing='ij')
    projected_r = np.hypot(xx, yy)
    for b, case in enumerate(cases):
        floor = max(.25, float(P.EDGES[:-1][case['fit_base_mask']].min())+
                    np.sqrt(2)*M.EXTENT/M.NPIX)
        for selection, components in SELECTIONS.items():
            for q, (field, (family, _)) in enumerate(FAMILIES.items()):
                for ri, entry in enumerate(case['selections'][selection][field]):
                    if not entry['fit'] or 'resolved_extrapolated_4R_map' in entry:
                        continue
                    radius = ri+2
                    pixels = (projected_r >= floor)&(projected_r < radius)
                    pixels4 = (projected_r >= floor)&(projected_r < 4)
                    truth = maps['map_sums'][b, 0, 1, ri, components, q].sum(axis=0)/case['halos']
                    truth4 = maps['map_sums'][b, 0, 1, 2, components, q].sum(axis=0)/case['halos']
                    model = project_map(family, entry['fit']['parameters'], radius)
                    model4 = project_map(family, entry['fit']['parameters'], 4)
                    entry['resolved_projection_min_b_R200c'] = floor
                    entry['resolved_projection_map'] = map_metrics(model[pixels], truth[pixels])
                    entry['resolved_extrapolated_4R_map'] = map_metrics(model4[pixels4], truth4[pixels4])
                    if entry.get('projected_refit'):
                        projected = project_map(family, entry['projected_refit']['parameters'], radius)
                        entry['resolved_projected_refit_map'] = map_metrics(projected[pixels], truth[pixels])
        (HERE/'products'/f'{case["label"]}_fits.json').write_text(json.dumps(plain(case), indent=2)+'\n')
    summary['additional_map_diagnostics'] = 'Core-excluded metrics derived from saved coefficients and maps; no refits.'
    summary_path.write_text(json.dumps(plain(summary), indent=2)+'\n')
    csv_path = HERE/'products/fit_comparison.csv'
    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames)
        table = list(reader)
    column = 'resolved_extrapolated_4R_map_L2_percent'
    if column not in columns:
        columns.append(column)
    by_low = {case['mass_edges'][0]: case for case in cases}
    for row in table:
        case = by_low[float(row['bin_low'])]
        entry = case['selections'][row['selection']][row['field']][int(float(row['radius_R200c']))-2]
        row[column] = entry['resolved_extrapolated_4R_map']['relative_L2_percent']
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    supported = [case for case in cases if case['strict_resolution_supported']]
    result = {'supported_bins': len(supported), 'diagnostic_bins': len(cases)-len(supported),
              'aggregation': 'Equal-bin medians; equal normalized-halo means within each bin.',
              'recommendation': 'Acquire 4R; fit 4R for an all-gas full-4R target. '
                  'For inner profiles use the 2R fit or a separate inner component. '
                  'Additive painting requires unique ownership and a diffuse component.'}
    for field in ('pressure', 'electrons'):
        valid = [case for case in supported if all(entry.get('accepted') and
            'heldout' in entry and accepted(entry['heldout']['fit'])
            for entry in case['selections']['all'][field])]
        errors = np.array([[entry['heldout']['all_4R_RMS_percent']
            for entry in case['selections']['all'][field]] for case in valid])
        inner = np.array([[entry['common_inner_RMS_percent']
            for entry in case['selections']['all'][field]] for case in supported])
        retention = np.array([case['integrated_fractions'][field]['retained_fraction_of_4R']
            for case in supported])
        components = np.array([case['integrated_fractions'][field]['component_fraction']
            for case in supported])
        ring = (components[:, 2] - components[:, 1]*retention[:, 1, None]) / (
            1-retention[:, 1, None])
        result[field] = {
            'stable_supported_holdout_bins': len(valid),
            'median_heldout_full_4R_RMS_percent': np.median(errors, axis=0),
            'heldout_full_4R_wins': np.bincount(np.argmin(errors, axis=1), minlength=3),
            'median_training_population_inner_2R_RMS_percent': np.median(inner, axis=0),
            'median_retained_fraction_of_4R': np.median(retention, axis=0),
            'median_cumulative_satellite_fraction': np.median(components[:, :, 1], axis=0),
            'median_cumulative_other_halo_fraction': np.median(components[:, :, 2], axis=0),
            'median_3_to_4R_ring_component_fraction': np.median(ring, axis=0),
            'median_all_4R_RMS_by_target': {selection: np.median([
                case['selections'][selection][field][2]['all_4R_RMS_percent']
                for case in supported]) for selection in ('all', 'no_other', 'no_satellites_or_other')},
            'median_outer_ring_map_L2_by_fit_radius_percent': {ring: np.median([
                [entry['outer_map_L2_percent'][ring]
                 for entry in case['selections']['all'][field]] for case in supported], axis=0)
                for ring in ('1to2', '2to3', '3to4')},
        }
    pressure = result['pressure']
    pressure['median_native_4R_map_L2_original_z_percent'] = np.median([
        case['selections']['all']['pressure'][2]['original_z_view_map']['relative_L2_percent']
        for case in supported])
    pressure['median_native_4R_map_L2_ten_views_percent'] = np.median([
        case['selections']['all']['pressure'][2]['matched_sphere_map']['relative_L2_percent']
        for case in supported])
    pressure['median_core_excluded_4R_map_L2_by_fit_radius_percent'] = np.median([
        [entry['resolved_extrapolated_4R_map']['relative_L2_percent']
         for entry in case['selections']['all']['pressure']] for case in supported], axis=0)
    result = plain(result)
    (HERE/'products/interpretation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
