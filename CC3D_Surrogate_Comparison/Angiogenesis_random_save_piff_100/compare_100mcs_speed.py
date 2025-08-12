'''
compares evaluation speed between 100 cc3d calls and 100 unet surrogate evaluations
'''
#%% imports
import os
import time
import torch
import numpy as np
import zarr
import evaluate_stats_functions as esf
from os.path import dirname, join, expanduser
from cc3d.CompuCellSetup.CC3DCaller import CC3DCaller
import gc
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt


def call_cc3d_sim(sim_fname, output_path):
    cc3d_sim_folder = 'Angiogenesis_random/Angiogenesis3.cc3d'
    workspace_path = os.path.join(output_path, 'workspace_dump')
    os.makedirs(workspace_path, exist_ok=True)
    cc3d_caller = CC3DCaller(
        cc3d_sim_fname=sim_fname,
        output_dir = workspace_path)
    ret_value = cc3d_caller.run()
    del cc3d_caller
    gc.collect()
    return

def rewrite_xml(xml_file_path, new_pif_name, num_processors=1):
    '''
    rewrites xml file with new pif name for loading in cc3d

    also allows for changing the number of processors if needed

    xml_file_path: str, path to xml file
    new_pif_name: str, new pif name to replace in xml file
    '''
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    pif_name_element = root.find('.//PIFName')
    num_processors_element = root.find('.//NumberOfProcessors')

    if num_processors_element is not None:
        num_processors_element.text = str(num_processors)
    else:
        print('NumberOfProcessors element not found in XML. There must be an error')


    if pif_name_element is not None:
        #replace with new string
        pif_name_element.text = new_pif_name

        tree.write(xml_file_path)
    else:
        print('<PIFName> element not found in the XML file.')

def load_and_stack_for_model_eval(zarr_file):
    '''Load the data from the zipstore at the given timestep index and at index + 10 for the 10 timesteps ahead'''
    with zarr.open(store=zarr.storage.ZipStore(zipstore_path, mode="r")) as root:
        fgbg = np.array(root["fgbg"][:])  
        vegf = np.array(root["vegf"][:])
        input_stack = np.stack([fgbg, vegf], axis=-1)
    return input_stack
    
def pass_through_model_time_on_device(model, input_stack,threshold=0.5,probabilities=False, do_warmup=False):
    '''Pass the input stack through the model and return the output, sigmoided and thresholded and time'''
    input_tensor = torch.from_numpy(input_stack).float().unsqueeze(0).permute(0, 3, 1, 2) #convert to tensor with shape (1, 2, 256, 256), (batch, channels, height, width)

    # Move the input tensor to the same device as the model
    input_tensor = input_tensor.to(model.device)

    # Ensure the model is in evaluation mode
    model.eval()

    # Disable gradient calculations
    with torch.no_grad():
        # Pass the input tensor through the model

        if do_warmup:
            #warmup the model by passing a dummy tensor through it
            # forces the model to load to the device and initialize any necessary resources and computations
            _ = model(input_tensor)

        time_on_device_start = time.perf_counter()
        output = model(input_tensor) # output is [batch_size, 2, 256, 256]
        time_on_device_eval = time.perf_counter() - time_on_device_start #time taken for evaluation on device

    #sigmoid the cell (segmentation) channel
    if probabilities:
        output[:,0,:,:] = torch.sigmoid(output[:,0,:,:]) #sigmoided for probabilities
    else:
        output[:,0,:,:] = torch.sigmoid(output[:,0,:,:])
        output[:,0,:,:] = (output[:,0,:,:]>threshold).int() #thresholded probabilities to binary mask

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy() #convert back to numpy with shape (256, 256, 2)
    return output, time_on_device_eval


evaluate_100mcs_folder = 'evaluate_100mcs_cc3d'
dir_up_for_workspace_dump = '' #CC3D must generate workspace files, create a folder for this and delete later.
zarr_folder = r'Angiogenesis_random_save_piff_100\evaluate_100mcs_cc3d\Simulation\zarr_files'
model_checkpoint_folder = r'' #trained model checkpoint path
model_checkpoint = esf.return_best_model(model_checkpoint_folder)
model = esf.load_model(model_checkpoint, 2, split_periodic=False)
model_cpu = esf.load_model(model_checkpoint, 2, split_periodic=False, device='cpu') #surrogate on CPU for CPU/CPU comparison

sim_files_folder = os.path.join(evaluate_100mcs_folder, 'Simulation')
xml_file_path = os.path.join(sim_files_folder, 'Angiogenesis.xml')
cc3d_sim_path = os.path.join(evaluate_100mcs_folder, 'Angiogenesis3.cc3d')

#%% model evaluations zarr for unet surrogate on GPU
evaluations_dict = {}

print('unet model evaluations gpu')
unet_model_evaluation_times = []
for i, zarr_file in enumerate(os.listdir(zarr_folder)):
    if zarr_file.endswith('.zarr.zip'):
        if i == 0:
            do_warmup = True
        else:
            do_warmup = False
        zipstore_path = os.path.join(zarr_folder, zarr_file)
        input_stack = load_and_stack_for_model_eval(zipstore_path)
        output, eval_time = pass_through_model_time_on_device(model, input_stack, do_warmup=do_warmup)
        print('time taken for model evaluation:', eval_time)
        unet_model_evaluation_times.append(eval_time)

print('done with unet model evaluations')
print('mean time for unet model evaluations:', np.mean(unet_model_evaluation_times))
print('std time for unet model evaluations:', np.std(unet_model_evaluation_times))

evaluations_dict['unet_model_evaluations'] = unet_model_evaluation_times

#%% cc3d evaluations CC3D native code on CPU, 1 processor
print('cc3d evaluations 1 processor CPU')
cc3d_evaluation_times = []
for file in os.listdir(evaluate_100mcs_folder):
    if file.endswith('.piff'):
        piff_file = file
        rewrite_xml(xml_file_path, piff_file)
        start_time = time.perf_counter()
        call_cc3d_sim(cc3d_sim_path, dir_up_for_workspace_dump)
        end_time = time.perf_counter()
        cc3d_evaluation_times.append(end_time - start_time)


print('done with cc3d evaluations 1 processor CPU')
print('mean time for cc3d evaluations:', np.mean(cc3d_evaluation_times))
print('std time for cc3d evaluations:', np.std(cc3d_evaluation_times))

print('mean time for unet model evaluations:', np.mean(unet_model_evaluation_times))
print('std time for unet model evaluations:', np.std(unet_model_evaluation_times))

evaluations_dict['cc3d_evaluations'] = cc3d_evaluation_times

#%% model evaluations zarr for unet surrogate on CPU
print('unet model evaluations on CPU')
unet_model_evaluation_times_cpu = []
for i, zarr_file in enumerate(os.listdir(zarr_folder)):
    if zarr_file.endswith('.zarr.zip'):
        if i == 0:
            do_warmup = True
        else:
            do_warmup = False
        zipstore_path = os.path.join(zarr_folder, zarr_file)
        input_stack = load_and_stack_for_model_eval(zipstore_path)
        output, eval_time = pass_through_model_time_on_device(model_cpu, input_stack, do_warmup=do_warmup)
        print('time taken for model evaluation on CPU:', eval_time)
        unet_model_evaluation_times_cpu.append(eval_time)

#%% cc3d evaluations on CPU, all processors CC3D native code
print('cc3d evaluations on all processors')
num_processors = os.cpu_count()  # Get the number of available processors
print(f'Using {num_processors} processors for CC3D evaluations')
cc3d_evaluation_times_all = []
for file in os.listdir(evaluate_100mcs_folder):
    if file.endswith('.piff'):
        piff_file = file
        rewrite_xml(xml_file_path, piff_file, num_processors=num_processors)  # Change to the number of processors you want to use
        start_time = time.perf_counter()
        call_cc3d_sim(cc3d_sim_path, dir_up_for_workspace_dump)
        end_time = time.perf_counter()
        cc3d_evaluation_times_all.append(end_time - start_time)

print('mean time for cc3d evaluations on all processors:', np.mean(cc3d_evaluation_times_all))
print('std time for cc3d evaluations on all processors:', np.std(cc3d_evaluation_times_all))

print('mean time for unet model evaluations on CPU:', np.mean(unet_model_evaluation_times_cpu))
print('std time for unet model evaluations on CPU:', np.std(unet_model_evaluation_times_cpu))
evaluations_dict['unet_model_evaluations_cpu'] = unet_model_evaluation_times_cpu
evaluations_dict['cc3d_evaluations_all_processors'] = cc3d_evaluation_times_all
#save evaluations dict to csv with pandas
evaluations_df = pd.DataFrame(evaluations_dict)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder_with_timestamp = os.path.join(evaluate_100mcs_folder, f'evaluations_comparison_cc3d_surrogate_{timestamp}')
os.makedirs(folder_with_timestamp, exist_ok=True)
evaluations_df.to_csv(os.path.join(folder_with_timestamp, 'evaluations_comparison_cc3d_surrogate.csv'))
print('evaluations saved to csv')
print('evaluations saved to folder:', folder_with_timestamp)

data_filepath = os.path.join(folder_with_timestamp, 'evaluations_comparison_cc3d_surrogate.csv')

# %% post reviewer changes - used perf_counter instead of time.time for more accurate timing
#  did comparisons with 1 vs 32 cores for CPU CC3D
#  cpu AI surrogate 

plt.rcParams['font.family'] = 'Arial' 
evaluations_file = data_filepath
evaluations_df = pd.read_csv(evaluations_file)

for col in evaluations_df.columns:
    print('mean time for', col, ':', evaluations_df[col].median())
    print('std time for', col, ':', evaluations_df[col].std())

title_size = 30
ylabel_size = 35
xlabel_size = 25
tick_size = 30
bar_color = 'none'
cmap = plt.get_cmap('Accent')

dot_color = [cmap(0), cmap(2), cmap(4), cmap(6)] 
dot_edgecolors = 'black'
dot_size = 50
error_color = 'black'
alpha_datapoints = 0.8 

# Exclude the index column
evaluations_df_no_index = evaluations_df.iloc[:, 1:]

evaluations_df_no_index.columns = ['Surrogate\nGPU', 'CC3D\nCPU 1 core', 'Surrogate\nCPU 32 cores', 'CC3D\nCPU 32 cores']

means = evaluations_df_no_index.mean()
standard_errors = evaluations_df_no_index.std()

plt.figure(figsize=(14, 10),dpi=300)

lower_error = np.zeros_like(standard_errors)
upper_error = standard_errors
asymmetric_error = [lower_error, upper_error]

bars = plt.bar(
    evaluations_df_no_index.columns,
    means,
    yerr=asymmetric_error,
    capsize=5,
    color=bar_color,
    edgecolor='black',
    linewidth=1.5, 
    error_kw={'elinewidth': 3.5, 'ecolor': error_color},
    width=0.5,
    zorder=10,
)

for i, col in enumerate(evaluations_df_no_index.columns):
    jittered_x = np.ones(len(evaluations_df_no_index[col])) * i + np.random.uniform(-0.1, 0.1, len(evaluations_df_no_index[col]))
    plt.scatter(
        jittered_x,
        evaluations_df_no_index[col],
        facecolors=dot_color[i], 
        edgecolors=dot_edgecolors, 
        alpha=alpha_datapoints,
        zorder=5,
        s=dot_size
    )

plt.ylabel('Evaluation Time (s)', fontsize=ylabel_size)
plt.xticks(fontsize=xlabel_size)
plt.yticks(fontsize=tick_size)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Set y-axis to log scale
plt.yscale('log')
plt.show()

# %% summary statistics
summary_data = []
for col in evaluations_df.columns:
    if col != 'Unnamed: 0':  # Skip the index column
        median_time = evaluations_df[col].median()
        std_time = evaluations_df[col].std()
        summary_data.append({
            'Method': col,
            'Median Time (s)': f'{median_time:.6f}',
            'Std Time (s)': f'{std_time:.6f}',
        })
# %% do stats
import scipy.stats as stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np

evaluations_df_no_index = evaluations_df.iloc[:, 1:]
evaluations_df_no_index.columns = ['Surrogate_GPU', 'CC3D_CPU_1core', 'Surrogate_CPU_32cores', 'CC3D_CPU_32cores']

data_long = []
for col in evaluations_df_no_index.columns:
    for value in evaluations_df_no_index[col]:
        data_long.append({'Method': col, 'Time': value})

df_long = pd.DataFrame(data_long)

f_stat, p_value = f_oneway(*[evaluations_df_no_index[col] for col in evaluations_df_no_index.columns])

print("=== ONE-WAY ANOVA RESULTS ===")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# Post-hoc testing (Tukey's HSD)
if p_value < 0.05:
    tukey_results = pairwise_tukeyhsd(endog=df_long['Time'], 
                                     groups=df_long['Method'], 
                                     alpha=0.05)
    print(tukey_results)


    
    # Summary table
    tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], 
                           columns=tukey_results._results_table.data[0])
    tukey_df['Significant'] = tukey_df['reject'].map({True: 'Yes', False: 'No'})
    print(tukey_df[['group1', 'group2', 'meandiff', 'p-adj', 'Significant']].to_string(index=False).replace("_", " ").replace("1core", "1 core").replace("32cores", "32 cores"))

    #scientific notation
    def format_pvalue(p):
        if p < 0.001:
            return f"{p:.2e}"
        else:
            return f"{p:.4f}"
    
    tukey_df['p-adj_formatted'] = tukey_df['p-adj'].apply(format_pvalue)
    
    display_cols = ['group1', 'group2', 'meandiff', 'p-adj_formatted', 'Significant']
    table_string = tukey_df[display_cols].to_string(index=False).replace("_", " ").replace("1core", "1 core").replace("32cores", "32 cores")
    table_string = table_string.replace("p-adj_formatted", "p-adj")
    print(table_string)