import streamlit as st
import flowkit as fk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import pandas as pd
import openpyxl
import os


# Define transformations
transformations = {
    'CD45': fk.transforms.WSPBiexTransform(
        'biex',
        max_value=262144.000029,
        positive=3.94,
        width=-1,
        negative=0
    ),
}
transformations2 = {
    'FSC-area': fk.transforms.WSPBiexTransform(
        'biex2',
        max_value=262144.000029,
        positive=4.418540,
        width=-10,
        negative=0
    ),
    'SigResidual': fk.transforms.WSPBiexTransform(
        'biex3',
        max_value=262144.000029,
        positive=4.418540,
        width=-10,
        negative=0
    ),
    'FSC-area': fk.transforms.LogicleTransform(
        'logicle',
        param_t=262144,
        param_w=0.5,
        param_m=4.5,
        param_a=0
    ),
    'SigResidual': fk.transforms.LogicleTransform(
        'logicle',
        param_t=262144,
        param_w=0.5,
        param_m=4.5,
        param_a=0
    ),
}
transformations3 = {
    'CD45': fk.transforms.LogicleTransform(
        'logicle',
        param_t=2621440000,
        param_w=1,
        param_m=4.5,
        param_a=10
    ),
}
transformations4 = {
    'Viability': fk.transforms.WSPBiexTransform(
        'biex3',
        max_value=262144.000029,
        positive=4.418540,
        width=-10,
        negative=0
    ),
    'Viability': fk.transforms.LogicleTransform(
        'logicle',
        param_t=262144,
        param_w=0.5,
        param_m=4.5,
        param_a=0
    ),
}
transformations5 = {
    'CD3': fk.transforms.WSPBiexTransform(
        'biex3',
        max_value=262144.000029,
        positive=4.418540,
        width=-10,
        negative=0
    ),
    'CD3': fk.transforms.LogicleTransform(
        'logicle',
        param_t=262144,
        param_w=0.5,
        param_m=4.5,
        param_a=0
    ),
}
transformations6 = {
    'CD4': fk.transforms.WSPBiexTransform(
        'biex3',
        max_value=262144.000029,
        positive=4.418540,
        width=-10,
        negative=0
    ),
    'CD4': fk.transforms.LogicleTransform(
        'logicle',
        param_t=262144,
        param_w=0.5,
        param_m=4.5,
        param_a=0
    ),
    'CD8': fk.transforms.WSPBiexTransform(
        'biex3',
        max_value=262144.000029,
        positive=4.418540,
        width=-10,
        negative=0
    ),
    'CD8': fk.transforms.LogicleTransform(
        'logicle',
        param_t=262144,
        param_w=0.5,
        param_m=4.5,
        param_a=0
    ),
}

def plot_with_rectangle(ax, x, y, title, xlabel, ylabel, rect_coords=None):
    ax.scatter(x, y, s=5, color="green", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rect_coords:
        left, right, bottom, top = rect_coords
        rect = Rectangle((left, bottom), right-left, top-bottom, linewidth=1, edgecolor='r', facecolor='none', alpha=0.1)
        ax.add_patch(rect)

def process_fcs_file_SLM(fcs_files):
    df_reports = pd.DataFrame()
    df_all_reports = pd.DataFrame()

    fig, axes = plt.subplots(len(fcs_files), 6, figsize=(30, 5 * len(fcs_files)))
    axes = axes.reshape(len(fcs_files), 6)  # Ensure axes is 2D

    for i, fcs_file in enumerate(fcs_files):
        sample = fk.Sample(fcs_file)
        
        sample.apply_transform(transformations, include_scatter=True)

        # Convert the transformed sample to a dataframe
        df_events_transformed = sample.as_dataframe(source='xform')

        # Create GatingStrategy
        g_strat = fk.GatingStrategy()

        # Gate 1: PeakTime and CD45
        dim_a = fk.Dimension('PeakTime', range_max=4e7, range_min=0)
        dim_b = fk.Dimension('CD45', range_min=100, range_max=1e5)
        rect_top_left_gate = fk.gates.RectangleGate('CD45sub', dimensions=[dim_a, dim_b])
        g_strat.add_gate(rect_top_left_gate, gate_path=('root',))

        # Apply gating strategy to the sample and store the results in `res`
        res = g_strat.gate_sample(sample)

        plot_with_rectangle(axes[i, 0], df_events_transformed['PeakTime'], df_events_transformed['CD45'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'PeakTime', 'CD45', 
                            rect_coords=(0, 4e7, 100, 1e5))

        gated_populations_gate1 = {}
        gate_id = g_strat.get_gate_ids()[0]
        gate_name = gate_id[0]
        gate_path = gate_id[1]
        membership = res.get_gate_membership(gate_name, gate_path)
        gated_populations_gate1[gate_name] = df_events_transformed[membership]

        sample2 = fk.Sample(gated_populations_gate1['CD45sub'], sample_id=f'{fcs_file}_g1')
        sample2.apply_transform(transformations2)
        df_events_transformed2 = sample2.as_dataframe(source='xform').sample(frac=0.3, random_state=42)
        sample2_from_df_transformed = fk.Sample(df_events_transformed2, sample_id=f'{fcs_file}_g1')

        g_strat2 = fk.GatingStrategy()
        # Gate 2: FSC-area and SigResidual
        dim_c = fk.Dimension('FSC-area', range_min=-0.5, range_max=1)
        dim_d = fk.Dimension('SigResidual', range_min=-1, range_max=1)
        rect_gate_2 = fk.gates.RectangleGate('FSC_SigResidual_gate', dimensions=[dim_c, dim_d])
        g_strat2.add_gate(rect_gate_2, gate_path=('root',))

        res2 = g_strat2.gate_sample(sample2_from_df_transformed)

        plot_with_rectangle(axes[i, 1], df_events_transformed2['FSC-area'], df_events_transformed2['SigResidual'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'FSC-area', 'SigResidual', 
                            rect_coords=(-0.5, 1, -1, 1))

        gated_populations_gate2 = {}
        gate_id1 = g_strat2.get_gate_ids()[0]
        gate_name1 = gate_id1[0]
        gate_path1 = gate_id1[1]
        membership2 = res2.get_gate_membership(gate_name1, gate_path1)
        gated_populations_gate2[gate_name1] = df_events_transformed2[membership2]

        sample3 = fk.Sample(gated_populations_gate2['FSC_SigResidual_gate'], sample_id=f'{fcs_file}_g2')
        sample3.apply_transform(transformations3)
        df_events_transformed3 = sample3.as_dataframe(source='xform')
        sample3_from_df_transformed = fk.Sample(df_events_transformed3, sample_id=f'{fcs_file}_g2')

        g_strat3 = fk.GatingStrategy()
        # Gate 3: FSC-area and CD45
        dim_e = fk.Dimension('FSC-area', range_min=0.2, range_max=0.7)
        dim_f = fk.Dimension('CD45', range_min=0.758635, range_max=0.758645)
        rect_gate_3 = fk.gates.RectangleGate('FSC_CD45_gate', dimensions=[dim_e, dim_f])
        g_strat3.add_gate(rect_gate_3, gate_path=('root',))

        res3 = g_strat3.gate_sample(sample3_from_df_transformed)

        plot_with_rectangle(axes[i, 2], df_events_transformed3['FSC-area'], df_events_transformed3['CD45'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'FSC-area', 'CD45', 
                            rect_coords=(0.2, 0.7, 0.758635, 0.758645))

        gated_populations_gate3 = {}
        gate_id2 = g_strat3.get_gate_ids()[0]
        gate_name2 = gate_id2[0]
        gate_path2 = gate_id2[1]
        membership3 = res3.get_gate_membership(gate_name2, gate_path2)
        gated_populations_gate3[gate_name2] = df_events_transformed3[membership3]

        sample4 = fk.Sample(gated_populations_gate3['FSC_CD45_gate'], sample_id=f'{fcs_file}_g3')
        sample4.apply_transform(transformations4)
        df_events_transformed4 = sample4.as_dataframe(source='xform')
        sample4_from_df_transformed = fk.Sample(df_events_transformed4, sample_id=f'{fcs_file}_g3')

        g_strat4 = fk.GatingStrategy()
        # Gate 4: FSC-area and Viability
        dim_g = fk.Dimension('FSC-area', range_min=0.2, range_max=0.6)
        dim_h = fk.Dimension('Viability', range_min=-0.3, range_max=0.6)
        rect_gate_4 = fk.gates.RectangleGate('FSC_Viability_gate', dimensions=[dim_g, dim_h])
        g_strat4.add_gate(rect_gate_4, gate_path=('root',))

        res4 = g_strat4.gate_sample(sample4_from_df_transformed)

        plot_with_rectangle(axes[i, 3], df_events_transformed4['FSC-area'], df_events_transformed4['Viability'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'FSC-area', 'Viability', 
                            rect_coords=(0.2, 0.6, -0.3, 0.6))

        gated_populations_gate4 = {}
        gate_id3 = g_strat4.get_gate_ids()[0]
        gate_name3 = gate_id3[0]
        gate_path3 = gate_id3[1]
        membership4 = res4.get_gate_membership(gate_name3, gate_path3)
        gated_populations_gate4[gate_name3] = df_events_transformed4[membership4]

        sample5 = fk.Sample(gated_populations_gate4['FSC_Viability_gate'], sample_id=f'{fcs_file}_g4')
        sample5.apply_transform(transformations5)
        df_events_transformed5 = sample5.as_dataframe(source='xform')
        sample5_from_df_transformed = fk.Sample(df_events_transformed5, sample_id=f'{fcs_file}_g4')

        g_strat5 = fk.GatingStrategy()
        # Gate 5: FSC-area and CD3
        dim_i = fk.Dimension('FSC-area', range_min=0.2, range_max=0.6)
        dim_j = fk.Dimension('CD3', range_min=0.5, range_max=0.8)
        rect_gate_5 = fk.gates.RectangleGate('FSC_CD3_gate', dimensions=[dim_i, dim_j])
        g_strat5.add_gate(rect_gate_5, gate_path=('root',))

        res5 = g_strat5.gate_sample(sample5_from_df_transformed)

        plot_with_rectangle(axes[i, 4], df_events_transformed5['FSC-area'], df_events_transformed5['CD3'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'FSC-area', 'CD3', 
                            rect_coords=(0.2, 0.6, 0.5, 0.8))

        gated_populations_gate5 = {}
        gate_id4 = g_strat5.get_gate_ids()[0]
        gate_name4 = gate_id4[0]
        gate_path4 = gate_id4[1]
        membership5 = res5.get_gate_membership(gate_name4, gate_path4)
        gated_populations_gate5[gate_name4] = df_events_transformed5[membership5]

        sample6 = fk.Sample(gated_populations_gate5['FSC_CD3_gate'], sample_id=f'{fcs_file}_g5')
        sample6.apply_transform(transformations6)
        df_events_transformed6 = sample6.as_dataframe(source='xform')
        sample6_from_df_transformed = fk.Sample(df_events_transformed6, sample_id=f'{fcs_file}_g5')

        g_strat6 = fk.GatingStrategy()
        # Gate 6: CD4 and CD8
        dim_k = fk.Dimension('CD4', range_min=0.15, range_max=0.8)
        dim_l = fk.Dimension('CD8', range_min=-0.4, range_max=0.5)
        rect_gate_6 = fk.gates.RectangleGate('CD4_gate', dimensions=[dim_k, dim_l])
        g_strat6.add_gate(rect_gate_6, gate_path=('root',))

        dim_m = fk.Dimension('CD4', range_min=-0.5, range_max=0.15)
        dim_n = fk.Dimension('CD8', range_min=0.5, range_max=0.8)
        rect_gate_6 = fk.gates.RectangleGate('CD8_gate', dimensions=[dim_m, dim_n])
        g_strat6.add_gate(rect_gate_6, gate_path=('root',))

        res6 = g_strat6.gate_sample(sample6_from_df_transformed)

        plot_with_rectangle(axes[i, 5], df_events_transformed6['CD4'], df_events_transformed6['CD8'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'CD4', 'CD8', 
                            rect_coords=(0.15, 0.8, -0.4, 0.5))
        plot_with_rectangle(axes[i, 5], df_events_transformed6['CD4'], df_events_transformed6['CD8'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'CD4', 'CD8', 
                            rect_coords=(-0.5, 0.15, 0.5, 0.8))

        reports = []

        # Append the report from 'res6' to the list
        reports.append(pd.DataFrame(res6.report))

        # Append 'df_all_reports' to 'reports'
        reports.append(df_all_reports)

        # Concatenate the DataFrames in 'reports'
        df_all_reports = pd.concat(reports, ignore_index=True)
        df_all_reports['gate_strategy'] = 'Statistical Learning Model'

    plt.tight_layout()
    st.pyplot(fig)
    return df_all_reports

