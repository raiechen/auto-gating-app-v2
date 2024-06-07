import streamlit as st
import flowkit as fk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import os
import pandas as pd
from main3 import process_fcs_file_SLM
import tempfile

st.title("Day0 automated gating system")

uploaded_files = st.file_uploader("Choose fcs files", accept_multiple_files=True)

fcs_files = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        tmp_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(tmp_path, 'wb') as tmp:
            tmp.write(uploaded_file.getvalue())
        fcs_files.append(tmp_path)

df_reports = pd.DataFrame()
df_all_reports = pd.DataFrame()
unprocessed_files = []

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

if fcs_files:
    fig, axes = plt.subplots(len(fcs_files), 6, figsize=(30, 5 * len(fcs_files)))
    axes = axes.reshape(len(fcs_files), 6)  # Ensure axes is 2D

    for i, fcs_file in enumerate(fcs_files):
        sample = fk.Sample(fcs_file)

        # Apply transformations
        sample.apply_transform(transformations, include_scatter=True)

        # Convert the transformed sample to a dataframe
        df_events_transformed = sample.as_dataframe(source='xform')

        # Create GatingStrategy
        g_strat = fk.GatingStrategy()

        # Gate 1: PeakTime and CD45
        dim_a = fk.Dimension('PeakTime', range_max=4e7, range_min=2e6)
        dim_b = fk.Dimension('CD45', range_min=100, range_max=1e5)
        rect_top_left_gate = fk.gates.RectangleGate('CD45sub', dimensions=[dim_a, dim_b])
        g_strat.add_gate(rect_top_left_gate, gate_path=('root',))

        # Apply gating strategy to the sample and store the results in `res`
        res = g_strat.gate_sample(sample)

        plot_with_rectangle(axes[i, 0], df_events_transformed['PeakTime'], df_events_transformed['CD45'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'PeakTime', 'CD45', 
                            rect_coords=(2e6, 4e7, 100, 1e5))

        gated_populations_gate1 = {}
        gate_id = g_strat.get_gate_ids()[0]
        gate_name = gate_id[0]
        gate_path = gate_id[1]
        membership = res.get_gate_membership(gate_name, gate_path)
        gated_populations_gate1[gate_name] = df_events_transformed[membership]

        sample2 = fk.Sample(gated_populations_gate1['CD45sub'], sample_id='TS_g1')
        sample2.apply_transform(transformations2)
        df_events_transformed2 = sample2.as_dataframe(source='xform').sample(frac=0.3, random_state=42)
        sample2_from_df_transformed = fk.Sample(df_events_transformed2, sample_id='TS_g1', subsample=10000)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.06, min_samples=100).fit(df_events_transformed2[['FSC-area', 'SigResidual']])
        labels = dbscan.labels_

        plot_with_rectangle(axes[i, 1], df_events_transformed2['FSC-area'], df_events_transformed2['SigResidual'], 
                            f"{fcs_file} Gated Population with DBSCAN Clusters", 'FSC-area', 'SigResidual')

        g_strat2 = fk.GatingStrategy()
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        for j in range(0, max(labels)+1):
            cluster_df = df_events_transformed2[labels == j]
            dim_a = fk.Dimension('FSC-area', range_max=cluster_df['FSC-area'].max(), range_min=cluster_df['FSC-area'].min())
            dim_b = fk.Dimension('SigResidual', range_max=cluster_df['SigResidual'].max(), range_min=cluster_df['SigResidual'].min())
            gate = fk.gates.RectangleGate(f'FSC-Sig-cluster_{j}', dimensions=[dim_a, dim_b])
            axes[i, 1].scatter(cluster_df['FSC-area'], cluster_df['SigResidual'], 
                                s=5, color=colors[j % len(colors)], alpha=0.5, label=f'SigResidual-Cluster {j}')
            rect = Rectangle((cluster_df['FSC-area'].min(), cluster_df['SigResidual'].min()), 
                             cluster_df['FSC-area'].max()-cluster_df['FSC-area'].min(), 
                             cluster_df['SigResidual'].max()-cluster_df['SigResidual'].min(), 
                             linewidth=1, edgecolor=colors[j % len(colors)], facecolor='none', alpha=0.1)
            axes[i, 1].add_patch(rect)
            g_strat2.add_gate(gate, gate_path=('root',))

        axes[i, 1].legend()

        res2 = g_strat2.gate_sample(sample2_from_df_transformed)

        gated_populations_gate2 = {}
        gate_id1 = g_strat2.get_gate_ids()[0]
        gate_name1 = gate_id1[0]
        gate_path1 = gate_id1[1]
        membership2 = res2.get_gate_membership(gate_name1, gate_path1)
        gated_populations_gate2[gate_name1] = df_events_transformed2[membership2]

        sample3 = fk.Sample(gated_populations_gate2['FSC-Sig-cluster_0'], sample_id='TS_g2')
        sample3.apply_transform(transformations3)
        df_events_transformed3 = sample3.as_dataframe(source='xform')
        sample3_from_df_transformed = fk.Sample(df_events_transformed3, sample_id='TS_g2')

        g_strat3 = fk.GatingStrategy()
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

        if df_events_transformed4[['FSC-area', 'Viability']].empty:
            unprocessed_files.append(fcs_file)
            print(f"No FSC-area and Viability events found in {fcs_file}. Skipping...")
            continue

        dbscan = DBSCAN(eps=0.04, min_samples=200).fit(df_events_transformed4[['FSC-area', 'Viability']])
        labels3 = dbscan.labels_

        plot_with_rectangle(axes[i, 3], df_events_transformed4['FSC-area'], df_events_transformed4['Viability'], 
                            f"{fcs_file} Gated Population with DBSCAN Clusters", 'FSC-area', 'Viability')

        g_strat4 = fk.GatingStrategy()
        for j in range(0, max(labels3)+1):
            cluster_df = df_events_transformed4[labels3 == j]
            dim_a = fk.Dimension('FSC-area', range_max=cluster_df['FSC-area'].max(), range_min=cluster_df['FSC-area'].min())
            dim_b = fk.Dimension('Viability', range_max=cluster_df['Viability'].max(), range_min=cluster_df['Viability'].min())
            gate = fk.gates.RectangleGate(f'FSC-Viability-cluster_{j}', dimensions=[dim_a, dim_b])
            axes[i, 3].scatter(cluster_df['FSC-area'], cluster_df['Viability'], 
                                s=5, color=colors[j % len(colors)], alpha=0.5, label=f'Viab-Cluster {j}')
            rect = Rectangle((cluster_df['FSC-area'].min(), cluster_df['Viability'].min()), 
                             cluster_df['FSC-area'].max()-cluster_df['FSC-area'].min(), 
                             cluster_df['Viability'].max()-cluster_df['Viability'].min(), 
                             linewidth=1, edgecolor=colors[j % len(colors)], facecolor='none', alpha=0.1)
            axes[i, 3].add_patch(rect)
            g_strat4.add_gate(gate, gate_path=('root',))

        axes[i, 3].legend()

        res4 = g_strat4.gate_sample(sample4_from_df_transformed)

        gated_populations_gate4 = {}
        gate_id3 = g_strat4.get_gate_ids()

        if not gate_id3:
            unprocessed_files.append(fcs_file)
            print(f"No gate of FSC-area and Viability {fcs_file} ids found. Skipping...")
            continue

        gate_id3 = g_strat4.get_gate_ids()[0]
        gate_name3 = gate_id3[0]
        gate_path3 = gate_id3[1]
        membership4 = res4.get_gate_membership(gate_name3, gate_path3)
        gated_populations_gate4[gate_name3] = df_events_transformed4[membership4]

        sample5 = fk.Sample(gated_populations_gate4['FSC-Viability-cluster_0'], sample_id=f'{fcs_file}_g4')
        sample5.apply_transform(transformations5)
        df_events_transformed5 = sample5.as_dataframe(source='xform')
        sample5_from_df_transformed = fk.Sample(df_events_transformed5, sample_id=f'{fcs_file}_g4')

        plot_with_rectangle(axes[i, 4], df_events_transformed5['FSC-area'], df_events_transformed5['CD3'], 
                            f"{fcs_file} Gated Population with DBSCAN Clusters", 'FSC-area', 'CD3')

        g_strat5 = fk.GatingStrategy()

        gmm = GaussianMixture(n_components=2, covariance_type='full').fit(df_events_transformed5[['FSC-area', 'CD3']])
        labels4 = gmm.predict(df_events_transformed5[['FSC-area', 'CD3']])

        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        median_cd3 = df_events_transformed5['CD3'].median()
        top_cluster_id = None

        for j in range(0, max(labels4)+1):
            cluster_df = df_events_transformed5[labels4 == j]
            dim_a = fk.Dimension('FSC-area', range_max=cluster_df['FSC-area'].max(), range_min=cluster_df['FSC-area'].min())
            dim_b = fk.Dimension('CD3', range_max=cluster_df['CD3'].max(), range_min=cluster_df['CD3'].min())
            gate = fk.gates.RectangleGate(f'FSC-CD3-cluster_{j}', dimensions=[dim_a, dim_b])
            axes[i, 4].scatter(cluster_df['FSC-area'], cluster_df['CD3'], 
                                s=5, color=colors[j % len(colors)], alpha=0.5, label=f'CD3-Cluster {j}')
            cluster_median_cd3 = cluster_df['CD3'].median()
            if cluster_median_cd3 > median_cd3 and top_cluster_id is None:
                top_cluster_id = j

            rect = Rectangle((cluster_df['FSC-area'].min(), cluster_df['CD3'].min()), 
                             cluster_df['FSC-area'].max()-cluster_df['FSC-area'].min(), 
                             cluster_df['CD3'].max()-cluster_df['CD3'].min(), 
                             linewidth=1, edgecolor=colors[j % len(colors)], facecolor='none', alpha=0.1)
            axes[i, 4].add_patch(rect)
            g_strat5.add_gate(gate, gate_path=('root',))

        axes[i, 4].legend()

        res5 = g_strat5.gate_sample(sample5_from_df_transformed)

        gated_populations_gate5 = {}
        gate_id4 = g_strat5.get_gate_ids()[top_cluster_id]
        gate_name4 = gate_id4[0]
        gate_path4 = gate_id4[1]
        membership5 = res5.get_gate_membership(gate_name4, gate_path4)
        gated_populations_gate5[gate_name4] = df_events_transformed5[membership5]

        sample6 = fk.Sample(gated_populations_gate5[f'FSC-CD3-cluster_{top_cluster_id}'], sample_id=f'{fcs_file}_g5')
        sample6.apply_transform(transformations6)
        df_events_transformed6 = sample6.as_dataframe(source='xform')
        sample6_from_df_transformed = fk.Sample(df_events_transformed6, sample_id=f'{fcs_file}_g5')

        plot_with_rectangle(axes[i, 5], df_events_transformed6['CD4'], df_events_transformed6['CD8'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'CD4', 'CD8', 
                            rect_coords=(0.15, 0.8, -0.4, 0.5))
        plot_with_rectangle(axes[i, 5], df_events_transformed6['CD4'], df_events_transformed6['CD8'], 
                            f"{fcs_file} Gated Population with Rectangle Gate", 'CD4', 'CD8', 
                            rect_coords=(-0.5, 0.15, 0.5, 0.8))

        g_strat6 = fk.GatingStrategy()
        dim_k = fk.Dimension('CD4', range_min=0.15, range_max=0.8)
        dim_l = fk.Dimension('CD8', range_min=-0.4, range_max=0.5)
        rect_gate_6 = fk.gates.RectangleGate('CD4_gate', dimensions=[dim_k, dim_l])
        g_strat6.add_gate(rect_gate_6, gate_path=('root',))

        dim_m = fk.Dimension('CD4', range_min=-0.5, range_max=0.15)
        dim_n = fk.Dimension('CD8', range_min=0.5, range_max=0.8)
        rect_gate_6 = fk.gates.RectangleGate('CD8_gate', dimensions=[dim_m, dim_n])
        g_strat6.add_gate(rect_gate_6, gate_path=('root',))

        res6 = g_strat6.gate_sample(sample6_from_df_transformed)

        reports = []

        reports.append(pd.DataFrame(res6.report))

        reports.append(df_all_reports)

        df_all_reports = pd.concat(reports, ignore_index=True)
        df_all_reports['gate_strategy'] = 'Hybrid-Supervised Learning Model'

    if unprocessed_files:
        df_process_SLM = process_fcs_file_SLM(unprocessed_files)
        df_all_reports = pd.concat([df_all_reports, df_process_SLM], ignore_index=True)

    df_all_reports = df_all_reports[['sample', 'gate_name', 'gate_type',
                                     'count', 'absolute_percent', 'gate_strategy']]

    st.dataframe(df_all_reports, width=5000)
    st.pyplot(fig)

# df_all_reports.to_excel("Day0-flow-data-all-final-report.xlsx", index=False)
