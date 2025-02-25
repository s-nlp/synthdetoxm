import matplotlib.pyplot as plt
import numpy as np
import json

# File names for the JSON data
file_names = {
    'German': 'de.json',
    'Spanish': 'es.json',
    'French': 'fr.json',
    'Russian': 'ru.json',
}

json_strings = {}
for language, filename in file_names.items():
    try:
        with open(filename, 'r') as f:
            json_strings[language] = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        exit()

model_mapping = {
    "Mistral Nemo": "Mistral-Nemo-Instruct-2407_",
    "Mistral Small": "Mistral-Small-Instruct-2409_",
    "Qwen 2.5 32B": "Qwen2_",
    "Command-R 0824": "c4ai-command-r-08-2024_",
    "Gemma 2 27B": "gemma-2-27b-it_",
    "Aya Expanse 8B": "aya-expanse-8b",
    "Aya Expanse 32B": "aya-expanse-32b",
    "Llama 3.1 70B (Lorablated)": "Llama-3_",  # Renamed to "Llama 3.1 (Lorablated)"
    "Llama 3 70B (Abliterated)": "Meta-Llama-3-70B-Instruct-abliterated-v3_"
}

languages = list(json_strings.keys())
model_names = list(model_mapping.keys())
data = {model: [0] * len(languages) for model in model_names} # Initialize data structure

for lang_idx, lang in enumerate(languages):
    lang_data = json.loads(json_strings[lang])
    for model_name_plot, model_name_prefix in model_mapping.items(): # Now model_name_prefix is directly a string
        for filename, value in lang_data.items():
            if model_name_prefix in filename: # Directly check if prefix is in filename
                data[model_name_plot][lang_idx] += value # Sum up values if filename matches prefix


# Plotting parameters
bar_width = 0.08 # Further reduced bar width to accommodate more bars
bar_positions = np.arange(len(languages))

fig, ax = plt.subplots(figsize=(14, 12)) # Slightly wider figure

# Extended color list to ensure enough colors
colors = ['#a6cee3', '#b2df8a', '#fdbf6f', '#ff7f00', '#cab2d6', '#33a0dc', '#e31a1c', '#fb9a99']
if len(model_names) > len(colors): # Handle case if more models than colors, though should be enough now
    num_needed_colors = len(model_names) - len(colors)
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis', num_needed_colors) # Use viridis colormap for extra colors
    extra_colors = [cmap(i) for i in range(num_needed_colors)]
    colors.extend(extra_colors)


for i, model in enumerate(model_names):
    ax.bar(bar_positions + i * bar_width, data[model], width=bar_width, label=model, color=colors[i], edgecolor='grey', alpha=0.7)

# Set labels and title
ax.set_xlabel('')
# ax.set_ylabel('Number of accepted samples', fontsize=40) # Increased y-axis label fontsize
# ax.set_title('Number of accepted samples w.r.t. LLM', fontsize=40) # Increased title fontsize
ax.set_xticks(bar_positions + (len(model_names)/2 - 0.5) * bar_width) # Centering x ticks, adjusted for more bars
ax.set_xticklabels(languages, fontsize=40) # Increased x-axis tick label fontsize
ax.legend(loc='upper left', fontsize=30) # Increased legend fontsize

# Set y-axis ticks and labels
yticks_positions = np.linspace(0, 4500, 4) # 4 evenly spaced ticks from 0 to 4500
ax.set_yticks(yticks_positions)
ax.set_yticklabels([f'{int(y)}' for y in yticks_positions], fontsize=30, rotation=45, ha='right') # Formatted as integers and increased fontsize
ax.set_ylim(0, 4500) # Adjusted y-axis limit to 4500

plt.tight_layout()
plt.savefig('num_samples_llms.pdf', bbox_inches='tight')
# plt.show()

