import os 
import re
import pickle
import numpy as np
import pandas as pd
import numpy.random as npr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io 
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import statsmodels.api as sm
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime

class load_behavioral_data:
    """
    A class for loading and processing behavioral data from experimental sessions.
    
    Class Attributes:
        visual_stim_maps (dict): Mapping of trial types to visual stimulus angles for different task types
        audio_stim_maps (dict): Mapping of trial types to audio stimulus presence for different task types
        context_maps (dict): Mapping of trial types to context values for different task types
        color_maps (dict): Mapping of trial types to colors for visualization
        color_maps_opto (dict): Mapping of trial types to colors for opto condition visualization
    """

    # Class-level mapping dictionaries
    visual_stim_maps = {
        'psych': {
            1: 0, 2: 15, 3: 25, 4: 65, 5: 75, 6: 90,
            7: 0, 8: 15, 9: 25, 10: 65, 11: 75, 12: 90,
            13: 0, 14: 15, 15: 25, 16: 65, 17: 75, 18: 90
        },
        '2loc': {
            1: 0, 2: 90, 3: 0, 4: 90, 5: 90, 6: 0
        },
        'aud': {
            1: np.nan, 2: np.nan
        },
        'aud_psych': {
            1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan,
            7: np.nan, 8: np.nan
        }
    }
    
    audio_stim_maps = {
        'psych': {
            1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1,
            7: 1, 8: 1, 9: 1, 10: 0, 11: 0, 12: 0,
            13: 1, 14: 1, 15: 1, 16: 0, 17: 0, 18: 0
        },
        '2loc': {
            1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1
        },
        'aud': {
            1: 0, 2: 1
        },
        'aud_psych': {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1,
            7: 1, 8: 1
        }
    }
    
    context_maps = {
        'psych': {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
            7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1,
            13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2
        },
        '2loc': {
            1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2
        },
        'aud': {
            1: 2, 2: 2
        },
        'aud_psych': {
            1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2,
            7: 2, 8: 2
        }
    }

    color_maps = {
        'psych': [
            'purple', 'purple', 'purple', 'purple', 'purple', 'purple',  # 1-6
            '#EC008C', '#EC008C', '#EC008C', '#EC008C', '#EC008C', '#EC008C',  # 7-12
            '#27AAE1', '#27AAE1', '#27AAE1', '#27AAE1', '#27AAE1', '#27AAE1'   # 13-18
        ],
        '2loc': [
            'purple', 'purple',  # 1-2
            '#EC008C', '#EC008C',  # 3-4
            '#27AAE1', '#27AAE1'   # 5-6
        ],
        'aud': ['#27AAE1'],
        'aud_psych': ['#27AAE1', '#27AAE1']
    }

    color_maps_opto = {
        'psych': [
            'purple',  # 1-6
            '#EC008C',  # 7-12
            '#27AAE1' # 13-18
        ],
        '2loc': [
            'purple',  # 1-2
            '#EC008C',  # 3-4
            '#27AAE1'  # 5-6
        ],
        'aud': ['#27AAE1', '#27AAE1'],
        'aud_psych': ['#27AAE1', '#27AAE1']
    }

    def __init__(self):
        pass

    def get_and_sort_dates(self, directory):
        """
        Retrieves and sorts dates from filenames in a specified directory.

        Args:
            directory (str): Path to directory containing data files

        Returns:
            tuple: Contains:
                - dates (list): Sorted list of datetime objects
                - sorted_six_digit_numbers (list): Sorted list of date strings in 'yymmdd' format
        """
        # Define the folder path
        folder_path = directory

        # Regular expression pattern to match six-digit numbers between underscores
        pattern = re.compile(r'_(\d{6})_')

        # List to store the six-digit numbers found
        six_digit_numbers = []

        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Search for six-digit numbers in the file names
                match = pattern.search(file)
                if match:
                    six_digit_numbers.append(match.group(1))  # Extract the matched number

        # Remove duplicates (if any)
        six_digit_numbers = list(set(six_digit_numbers))

        # Convert the six-digit numbers to datetime objects for sorting by month and day
        dates = [datetime.strptime(date, '%y%m%d') for date in six_digit_numbers]

        # Sort the dates by month and day, ignoring the year
        dates.sort(key=lambda x: (x.month, x.day))

        # Sort the dates from earliest to latest
        dates.sort()

        # Convert back to original six-digit format and store in a sorted list
        sorted_six_digit_numbers = [date.strftime('%y%m%d') for date in dates]

        return dates, sorted_six_digit_numbers

    def convert_array_choice(self, arr):
        """
        Converts choice array from left/right coding to 0/1 coding.

        Args:
            arr (list): Array of choices where 0=left, 1=right

        Returns:
            list: Converted array where each element is flipped (0->1, 1->0)
        """
        return [1 - x for x in arr]
    
    def structured_array_to_dict(self, arr):
        """
        Converts structured NumPy array to dictionary format.

        Args:
            arr (np.ndarray): Structured array to convert

        Returns:
            dict or list: Dictionary representation of array if named fields exist,
                         otherwise returns list of array values
        """
        if not arr.dtype.names:
            return arr.tolist()
        return {name: self.structured_array_to_dict(arr[name]) for name in arr.dtype.names}
    
    def convert_dataCell_to_dict(self, mat_Cell_data):
        """
        Converts MATLAB Cell data to Python dictionary format.

        Args:
            mat_Cell_data (dict): MATLAB Cell data loaded from .mat file

        Returns:
            list: List of dictionaries containing converted data from each cell
        """
        dataCell_list = [self.structured_array_to_dict(mat_Cell_data['dataCell'][0, i][0, 0]) for i in range(mat_Cell_data['dataCell'].shape[1])]
        return dataCell_list
    
    def get_perc_correct_all_conditions(self, dataCell_list):
        """
        Calculates performance metrics for all conditions in a session.

        Args:
            dataCell_list (list): List of dictionaries containing trial data

        Returns:
            tuple: Contains:
                - correct_values (list): Binary list of correct/incorrect trials
                - condition_values (list): List of condition numbers for each trial
                - total_counts (dict): Total trials per condition
                - correct_counts (dict): Correct trials per condition
                - percentage_correct (dict): Percentage correct per condition
                - sorted_conditions (list): Conditions sorted numerically
                - sorted_percentages (list): Percentage correct values sorted by condition
        """
        correct_values = [int(entry['result']['correct'][0][0][0]) for entry in dataCell_list]
        condition_values = [int(entry['maze']['condition'][0][0][0]) for entry in dataCell_list]
        total_counts = {}
        correct_counts = {}
        for condition, correct in zip(condition_values, correct_values):
            if condition not in total_counts:
                total_counts[condition] = 0
                correct_counts[condition] = 0
            total_counts[condition] += 1
            correct_counts[condition] += correct
        percentage_correct = {condition: (correct_counts[condition] / total_counts[condition]) * 100 for condition in total_counts}
        sorted_conditions = sorted(percentage_correct.keys())
        sorted_percentages = [percentage_correct[condition] for condition in sorted_conditions]
        return correct_values, condition_values, total_counts, correct_counts, percentage_correct, sorted_conditions, sorted_percentages
    
    def get_behavioral_data_for_session(self, mouse_id, directory, date):
        """
        Loads and processes behavioral data for a single experimental session.
        """
        # Ensure directory path ends with '/'
        directory = os.path.join(directory, '')
        
        # Use mouse_id directly since it already has the correct prefix
        Cell_pattern = f'{mouse_id}_{date}_Cell'
        try:
            print(f"\nProcessing mouse: {mouse_id}")
            print(f"Looking in directory: {directory}")
            print(f"Looking for pattern: {Cell_pattern}")
            
            # Check if directory exists
            if not os.path.exists(directory):
                print(f"Directory does not exist: {directory}")
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            all_files = os.listdir(directory)
            mat_files = [f for f in all_files if f.endswith('.mat')]
            print(f"Found {len(mat_files)} .mat files:")
            for f in mat_files:
                print(f"  {f}")
                
            Cell_files = [f for f in all_files if f.endswith('.mat') and Cell_pattern in f]
            if Cell_files:
                print(f"Matched files: {Cell_files}")
            
        except OSError as e:
            print(f"Error accessing directory {directory}: {e}")
            Cell_files = []

        if not Cell_files:
            raise FileNotFoundError(f"No data files found for mouse {mouse_id} on date {date}")
        
        # Sort files to ensure we combine them in sequence order
        Cell_files = sorted(Cell_files, key=lambda x: int(x.split('_Cell_')[-1].split('.')[0]) if '_Cell_' in x else -1)
        
        # Initialize an empty DataFrame to combine all data
        combined_df = pd.DataFrame()

        for file_name in Cell_files:
            data = scipy.io.loadmat(os.path.join(directory, file_name))
            dataCell_list = self.convert_dataCell_to_dict(data)

            # Extract trial types first to determine tasktype
            condition_values = [int(entry['maze']['condition'][0][0][0]) for entry in dataCell_list]
            unique_trial_types = len(set(condition_values))
            
            # Automatically determine tasktype based on number of unique trial types
            if unique_trial_types == 1:
                print(f"Warning: Only 1 trial type found for Mouse {mouse_id}, Date {date}. Assuming 'aud' task type.")
                tasktype = 'aud'
            elif unique_trial_types == 2:
                tasktype = 'aud'
            elif unique_trial_types == 8:
                tasktype = 'aud_psych'
            elif 2 < unique_trial_types < 7:
                tasktype = '2loc'
            elif unique_trial_types > 6:
                tasktype = 'psych'
            else:
                raise ValueError(f"Unexpected number of trial types: {unique_trial_types}")

            choice_inverted = np.array([int(entry['result']['leftTurn'][0][0][0]) for entry in dataCell_list])  # choices - coded as 1 for L and 0 for 1 - we will reverse this
            choice = self.convert_array_choice(choice_inverted)  # convert choice so that 1 = R and 0 = L
            correct_values, condition_values, total_counts, correct_counts, percentage_correct, sorted_conditions, sorted_percentages = self.get_perc_correct_all_conditions(dataCell_list)
            correct = np.array(correct_values)
            # Check if isOptoTrial exists and get opto values accordingly
            opto_values = [int(entry['maze'].get('isOptoTrial', [[[[0]]]])[0][0][0]) if 'isOptoTrial' in entry['maze'] 
                          else 0 for entry in dataCell_list]
            stim = np.array(opto_values)
            trial_type = np.array(condition_values)
            data = {
                'choice': choice,  # 0 = left, 1 = right
                'trial_type': trial_type,  # 1-6 V, 7-12 A
                'outcome': correct,  # 0 = incorrect, 1 = correct
                'Opto': stim  # 0 = no stim, 1 = stim
            }
            df = pd.DataFrame(data)
            
            df['Visual_Stim'] = df['trial_type'].map(self.visual_stim_maps[tasktype])
            df['Audio_Stim'] = df['trial_type'].map(self.audio_stim_maps[tasktype])
            df['context'] = df['trial_type'].map(self.context_maps[tasktype])
            df['task_type'] = tasktype  # Add task_type column

            # Append the current file's DataFrame to the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        return combined_df, sorted_conditions, sorted_percentages

    def get_data_multiple_sessions(self, experimenter, mouse_id, directory, dates):
        """
        Combines behavioral data from multiple sessions.

        Args:
            experimenter (str): Experimenter identifier
            mouse_id (str): Mouse identifier
            directory (str): Path to data directory
            dates (list): List of session dates in 'yymmdd' format

        Returns:
            pd.DataFrame: Combined data from all sessions with added date column
        """
        # Create an empty DataFrame to store all sessions' data
        combined_df = pd.DataFrame()
        
        for date in dates:
            # Load data for each session
            df_single_session, sorted_conditions, percentage_correct = self.get_behavioral_data_for_session(
                mouse_id, 
                directory, 
                date
            )
            
            # Add the 'date' column to the session data
            df_single_session['date'] = date

            # Reorder columns to make 'date' the first column
            cols = ['date'] + [col for col in df_single_session.columns if col != 'date']
            df_single_session = df_single_session[cols]
            
            # Append the session data to the combined DataFrame
            combined_df = pd.concat([combined_df, df_single_session], ignore_index=True)
        
        return combined_df
    
    def filter_dataframe_by_task_and_opto(self, df, task_type, opto_value):
        """
        Filters behavioral data by task type and optogenetic condition.

        Args:
            df (pd.DataFrame): Input behavioral data
            task_type (str): Task type ('aud', '2loc', or 'psych')
            opto_value (int): Optogenetic condition (0 or 1)

        Returns:
            pd.DataFrame: Filtered behavioral data
        """
        # Filter the DataFrame based on task_type and Opto value
        filtered_df = df[(df['task_type'] == task_type) & (df['Opto'] == opto_value)].reset_index(drop=True)
        return filtered_df
    
    def calculate_percent_correct_by_condition(self, df):
        """
        Calculates performance for each experimental condition.

        Args:
            df (pd.DataFrame): Behavioral data with 'trial_type' and 'outcome' columns

        Returns:
            tuple: Contains:
                - conditions (list): List of condition numbers
                - percent_correct_values (list): Corresponding performance percentages
        """
        # Group the DataFrame by 'trial_type' and calculate the mean of 'outcome' for each group
        percent_correct = df.groupby('trial_type')['outcome'].mean() * 100

        # Extract conditions and percent correct values as separate lists
        conditions = percent_correct.index.tolist()
        percent_correct_values = percent_correct.values.tolist()

        return conditions, percent_correct_values
    
    def calculate_avg_percent_correct_by_task(self, df, task_type):
        """
        Calculates average performance for condition groups based on task type.

        Args:
            df (pd.DataFrame): Behavioral data
            task_type (str): Task type ('aud', '2loc', or 'psych')

        Returns:
            list: Average performance percentages for each condition group
        """
        # Define condition groups based on task type
        condition_groups = {
            'aud': [(1, 2)],
            'aud_psych': [(1, 4), (5, 8)],
            '2loc': [(1, 2), (3, 4), (5, 6)],
            'psych': [(1, 6), (7, 12), (13, 18)]
        }

        # Get the relevant condition groups for the given task type
        groups = condition_groups.get(task_type, [])

        # Calculate average percentage correct for each group
        avg_percent_correct = []
        for group in groups:
            # Filter the DataFrame for the current group
            group_df = df[df['trial_type'].between(group[0], group[1])]
            
            # Calculate the mean of 'outcome' for the group
            avg_percent = group_df['outcome'].mean() * 100
            avg_percent_correct.append(avg_percent)

        return avg_percent_correct

    def get_behavior_data_single_session_all_animals(self, mouse_ids, date, base_directory='/Volumes/Runyan5/Akhil/behavior/'):
        """
        Retrieves behavioral data for multiple animals from a single session.
        """
        # Ensure base directory path ends with '/'
        base_directory = os.path.join(base_directory, '')
        print(f"Base directory: {base_directory}")
        
        # Initialize results dictionary
        results = {}
        
        # Process each mouse
        for mouse_id in mouse_ids:
            try:
                # Use mouse_id directly since it already has the correct prefix
                directory = os.path.join(base_directory, mouse_id)
                
                if not os.path.exists(directory):
                    print(f"Directory does not exist: {directory}")
                    raise FileNotFoundError(f"No directory found for mouse {mouse_id}")
                
                # Get session data
                df_single_session, conditions, percentage_correct = self.get_behavioral_data_for_session(
                    mouse_id, 
                    directory, 
                    date
                )
                
                # Store results
                results[mouse_id] = {
                    'df': df_single_session,
                    'conditions': conditions,
                    'percentage_correct': percentage_correct,
                    'error': None
                }
                
            except Exception as e:
                # Store error information if processing fails
                results[mouse_id] = {
                    'df': None,
                    'conditions': None,
                    'percentage_correct': None,
                    'error': str(e)
                }
                print(f"Error processing Mouse {mouse_id}: {str(e)}")
                
        return results
    
    def get_behavior_data_all_sessions_all_animals(self, mouse_ids, experimenter='CB', base_directory='/Volumes/Runyan5/Akhil/behavior/'):
        """
        Retrieves and combines behavioral data for multiple animals across all their sessions.

        Args:
            mouse_ids (list): List of mouse identifiers
            experimenter (str, optional): Experimenter identifier. Defaults to 'CB'.
            base_directory (str, optional): Base path to data directory. 
                Defaults to '/Volumes/Runyan5/Akhil/behavior/'.

        Returns:
            dict: Dictionary containing for each mouse_id:
                - df (pd.DataFrame): Combined behavioral data across sessions
                - dates (list): List of datetime objects for sessions
                - sorted_dates (list): List of session dates in 'yymmdd' format
                - error (str or None): Error message if processing failed
        """
        # Initialize results dictionary
        results = {}
        
        # Process each mouse
        for mouse_id in mouse_ids:
            try:
                # Construct the directory path
                directory = os.path.join(base_directory, f'{experimenter}{mouse_id}/')
                
                # Get and sort dates for this mouse
                dates, sorted_dates = self.get_and_sort_dates(directory)
                
                # Initialize an empty DataFrame to store all sessions' data for this mouse
                combined_df = pd.DataFrame()
                
                # Process each session date
                for date in sorted_dates:
                    try:
                        # Get session data
                        df_single_session, _, _ = self.get_behavioral_data_for_session(
                            mouse_id, 
                            directory, 
                            date
                        )
                        
                        # Add the 'date' column to the session data
                        df_single_session['date'] = date

                        # Reorder columns to make 'date' the first column
                        cols = ['date'] + [col for col in df_single_session.columns if col != 'date']
                        df_single_session = df_single_session[cols]
                        
                        # Append the session data to the combined DataFrame
                        combined_df = pd.concat([combined_df, df_single_session], ignore_index=True)
                    
                    except Exception as e:
                        print(f"Error processing Mouse {mouse_id}, Date {date}: {str(e)}")
                        continue
                
                # Store results
                results[mouse_id] = {
                    'df': combined_df,
                    'dates': dates,
                    'sorted_dates': sorted_dates,
                    'error': None
                }
                
            except Exception as e:
                # Store error information if processing fails
                results[mouse_id] = {
                    'df': None,
                    'dates': None,
                    'sorted_dates': None,
                    'error': str(e)
                }
                print(f"Error processing Mouse {mouse_id}: {str(e)}")
                
        return results
    
class plot_behavioral_data:
    """
    Class for visualizing behavioral data through various plotting functions.
    """

    def __init__(self):
        pass

    def plot_perc_correct_by_condition(self, sorted_conditions, sorted_percentages, tasktype, ax=None):
        """
        Creates bar plot of performance by condition.

        Args:
            sorted_conditions (list): Sorted condition numbers
            sorted_percentages (list): Corresponding performance percentages
            tasktype (str): Task type ('aud', '2loc', or 'psych')
            ax (matplotlib.axes.Axes, optional): Axes for plotting. Defaults to None.

        Returns:
            matplotlib.axes.Axes: The plot axes object
        """
        # Define color maps for each task type
        color_maps = load_behavioral_data.color_maps
        
        # Get the color map for the given task type
        colors = color_maps.get(tasktype, ['skyblue'])  # Default to 'skyblue' if tasktype is not found
        
        # Create new figure if ax is not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
            
        # Create the plot
        ax.bar(sorted_conditions, sorted_percentages, color=colors[:len(sorted_conditions)])
        ax.set_xlabel('Condition')
        ax.set_ylabel('Percentage Correct')
        ax.set_title('Percentage Correct by Condition')
        ax.set_xticks(sorted_conditions)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Only show if we created a new figure
        if ax is None:
            plt.show()
            
        return ax

    def plot_comparison_of_avg_percent_correct(self, avg_percent_correct_1, avg_percent_correct_2, labels, tasktype, title='Comparison of Average Percent Correct'):
        """
        Creates comparison plot of performance between two datasets.

        Args:
            avg_percent_correct_1 (list): Performance values for first dataset
            avg_percent_correct_2 (list): Performance values for second dataset
            labels (list): Labels for condition groups
            tasktype (str): Task type ('aud', '2loc', or 'psych')
            title (str, optional): Plot title. Defaults to 'Comparison of Average Percent Correct'

        Returns:
            None: Displays the plot
        """
        # Define color maps for each task type
        color_maps = load_behavioral_data.color_maps_opto
        
        # Get the color map for the given task type
        colors = color_maps.get(tasktype, ['skyblue'])  # Default to 'skyblue' if tasktype is not found

        # Ensure there are enough colors for the number of groups
        if len(colors) < len(labels):
            colors = colors * (len(labels) // len(colors) + 1)

        # Define the number of groups
        n_groups = len(avg_percent_correct_1)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 5))

        # Define the bar width
        bar_width = 0.35

        # Define the positions of the bars
        index = np.arange(n_groups)

        # Plot the bars for each dataset
        bar1 = ax.bar(index, avg_percent_correct_1, bar_width, label='Non-Opto', color=colors[:n_groups])
        bar2 = ax.bar(index + bar_width, avg_percent_correct_2, bar_width, label='Opto', color=colors[:n_groups], alpha=0.6)

        # Add labels, title, and legend
        ax.set_xlabel('Condition Groups')
        ax.set_ylabel('Average Percent Correct')
        ax.set_title(title)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.legend()

        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()
    
    def plot_behavior_single_session(self, mouse_ids, date, base_directory):
        """
        Creates performance plots for multiple mice in a single session.

        Args:
            mouse_ids (list): List of mouse identifiers
            date (str): Session date in 'yymmdd' format

        Returns:
            tuple: Contains:
                - fig (matplotlib.figure.Figure): The figure object
                - axes (numpy.ndarray): Array of axes objects
        """
        # Define condition ranges and colors for different task types
        condition_ranges = {
            'aud': {'A': (1, 2)},
            'aud_psych': {'A': (1, 8)},
            '2loc': {'C': (1, 3), 'V': (3, 5), 'A': (5, 7)},
            'psych': {'C': (1, 7), 'V': (7, 13), 'A': (13, 19)}
        }
        colors = {'C': 'purple', 'V': '#EC008C', 'A': '#27AAE1'}

        # Calculate number of columns
        n_cols = 3
        n_rows = int(np.ceil(len(mouse_ids) / n_cols))

        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten()

        # Create instance of load_behavioral_data
        load_data = load_behavioral_data()

        # Loop through each mouse ID
        for idx, mouse_id in enumerate(mouse_ids):
            try:
                # Construct the directory path and get data
                directory = base_directory + '{}/'.format(mouse_id)
                
                # Get session data
                df_single_session, conditions, percentage_correct = load_data.get_behavioral_data_for_session(
                    mouse_id, directory, date
                )
                
                # Calculate additional metrics
                num_trials = len(df_single_session)
                percent_correct = (df_single_session['outcome'] == 1).mean() * 100
                percent_left = (df_single_session['choice'] == 0).mean() * 100
                opto_performed = df_single_session['Opto'].any()
                task_type = df_single_session['task_type'][0]
                
                # Print additional information
                opto_status = "Opto" if opto_performed else "No Opto"
                print(f'Mouse: {mouse_id}, Date: {date}, Trials: {num_trials}, % Correct: {percent_correct:.1f}, {opto_status}')
                
                # Sort conditions and percentages together
                sorted_indices = np.argsort(conditions)
                sorted_conditions = np.array(conditions)[sorted_indices]
                sorted_percentages = np.array(percentage_correct)[sorted_indices]
                
                # Plot the data on the specific subplot
                ax = axes[idx]
                
                # Determine bar colors based on task type and condition ranges
                bar_colors = []
                for condition in sorted_conditions:
                    if task_type == 'aud' or task_type == 'aud_psych':
                        bar_colors.append(colors['A'])
                    else:
                        for cond_type, (start, end) in condition_ranges[task_type].items():
                            if start <= condition < end:
                                bar_colors.append(colors[cond_type])
                                break
                
                # Plot bars with appropriate colors
                bars = ax.bar(range(len(sorted_conditions)), sorted_percentages, color=bar_colors)
                
                # Customize plot
                ax.set_xticks(range(len(sorted_conditions)))
                ax.set_xticklabels(sorted_conditions)
                ax.set_ylim(-20, 100)  # Extended y-axis range to accommodate text
                ax.set_ylabel('Percent Correct')
                ax.set_title(f'Mouse: {mouse_id}\nDate: {date}')
                ax.grid(False)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom',
                           fontsize=8)
                
                # Add trial numbers at bottom
                ax.text(0.5, -10,
                        f"n={num_trials}\n{percent_left:.1f}% L",
                        ha='center', va='center',
                        transform=ax.transData,
                        fontsize=8)
            
            except Exception as e:
                print(f"Error processing Mouse: {mouse_id} - {str(e)}")
                continue

        # Remove empty subplots if any
        for idx in range(len(mouse_ids), n_rows * n_cols):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()
        
        return fig, axes

    def plot_behavior_across_sessions(self, mouse_ids, base_directory, verbose=True):
        """
        Creates performance plots showing trends across multiple sessions.

        Args:
            mouse_ids (list): List of mouse identifiers
            verbose (bool, optional): Whether to print detailed information. Defaults to True.

        Returns:
            tuple: Contains:
                - fig (matplotlib.figure.Figure): The figure object
                - axes (numpy.ndarray): Array of axes objects
        """
        # Define condition ranges and colors for different task types
        condition_ranges = {
            'aud': {'A': (1, 2)},
            'aud_psych': {'A': (1, 8)},
            '2loc': {'C': (1, 3), 'V': (3, 5), 'A': (5, 7)},
            'psych': {'C': (1, 7), 'V': (7, 13), 'A': (13, 19)}
        }
        colors = {'C': 'purple', 'V': '#EC008C', 'A': '#27AAE1'}

        # Calculate number of columns (fixed at 3) and required rows
        n_cols = 3
        n_rows = int(np.ceil(len(mouse_ids) / n_cols))

        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=200)
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        # Create instance of load_behavioral_data
        load_data = load_behavioral_data()

        # Loop through each mouse ID
        for idx, mouse_id in enumerate(mouse_ids):
            try:
                #directory = '/Volumes/Runyan5/Akhil/behavior/{}/'.format(mouse_id)
                directory = base_directory + '{}/'.format(mouse_id)
                # Get directory and dates for this mouse
                dates, sorted_dates = load_data.get_and_sort_dates(directory)
                
                # Get last 5 dates
                last_5_dates = sorted_dates[-5:]
                performance_data = []
                
                # Collect data for each date
                for date in last_5_dates:
                    try:
                        df_single_session, conditions, percentage_correct = load_data.get_behavioral_data_for_session(
                            mouse_id, directory, date
                        )
                        
                        task_type = df_single_session['task_type'][0]
                        
                        # Calculate metrics based on task type
                        if task_type == 'aud':
                            condition_data = {}
                            # For auditory, all trials are type 'A'
                            condition_data['A'] = {
                                'percent_correct': (df_single_session['outcome'] == 1).mean() * 100,
                                'num_trials': len(df_single_session)
                            }
                        elif task_type in ['2loc', 'psych']:
                            condition_data = {}
                            for cond_type, (start, end) in condition_ranges[task_type].items():
                                mask = (df_single_session['trial_type'] >= start) & (df_single_session['trial_type'] < end)
                                cond_trials = df_single_session[mask]
                                if len(cond_trials) > 0:
                                    condition_data[cond_type] = {
                                        'percent_correct': (cond_trials['outcome'] == 1).mean() * 100,
                                        'num_trials': len(cond_trials)
                                    }
                        
                        num_trials = len(df_single_session)
                        opto_performed = df_single_session['Opto'].any()
                        percent_left = (df_single_session['choice'] == 0).mean() * 100
                        
                        # Store data
                        performance_data.append({
                            'date': date,
                            'task_type': task_type,
                            'condition_data': condition_data,
                            'num_trials': num_trials,
                            'opto': opto_performed,
                            'percent_left': percent_left
                        })
                        
                        # Print information only if verbose is True
                        if verbose:
                            opto_status = "Opto" if opto_performed else "No Opto"
                            print(f'Mouse: {mouse_id}, Date: {date}, Task: {task_type}, Trials: {num_trials}, % Left: {percent_left:.1f}, {opto_status}')
                            for cond_type, data in condition_data.items():
                                print(f'  {cond_type}: {data["percent_correct"]:.1f}% correct (n={data["num_trials"]})')
                    
                    except Exception as e:
                        if verbose:
                            print(f"Error processing date {date}: {str(e)}")
                        continue
                
                # Create the plot
                ax = axes[idx]
                
                # Plot performance for each condition type
                for cond_type in ['C', 'V', 'A']:
                    dates_idx = []
                    perf_values = []
                    opto_status = []
                    
                    for i, data in enumerate(performance_data):
                        if data['condition_data'] and cond_type in data['condition_data']:
                            dates_idx.append(i)
                            perf_values.append(data['condition_data'][cond_type]['percent_correct'])
                            opto_status.append(data['opto'])
                    
                    if dates_idx:  # Only plot if we have data for this condition
                        # Plot line first
                        ax.plot(dates_idx, perf_values, '-', color=colors[cond_type], 
                               linewidth=2, alpha=1.0)
                        
                        # Plot points with varying alpha based on opto status
                        for i, (x, y, opto) in enumerate(zip(dates_idx, perf_values, opto_status)):
                            alpha = 0.5 if opto else 1.0
                            ax.plot(x, y, 'o', color=colors[cond_type], 
                                   label=cond_type if i == 0 else "", 
                                   linewidth=2, markersize=8, alpha=alpha)
                
                # Customize the plot
                ax.set_xlabel('Session')
                ax.set_ylabel('Percent Correct')
                ax.set_title(f'Mouse: {mouse_id}\nPerformance over last 5 sessions')
                ax.set_xticks(range(len(last_5_dates)))
                ax.set_xticklabels(last_5_dates, rotation=45)
                ax.set_ylim(-10, 100)  # Extended y-axis range to accommodate text
                ax.set_xlim(-0.5, len(last_5_dates) + .5)
                ax.grid(False)  # Remove grid lines
                ax.legend()
                
                # Add trial numbers
                for i, data in enumerate(performance_data): 
                    # Position the annotation at the bottom of the plot
                    ax.annotate(f"n={data['num_trials']}\n{data['percent_left']:.1f}% L", 
                            (i, -5),  # Position at y=-5
                            textcoords="data", 
                            ha='center',
                            fontsize=8)  # Smaller font size

            except Exception as e:
                if verbose:
                    print(f"Error processing Mouse: {mouse_id} - {str(e)}")
                continue

        # Remove empty subplots if any
        for idx in range(len(mouse_ids), n_rows * n_cols):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()
        
        return fig, axes

class psychometrics:
    """
    Class for analyzing and plotting psychometric data from behavioral experiments.
    """

    def __init__(self):
        pass

    def calculate_percent_right_choice(self, df):
        """
        Calculates percentage of right choices for each visual stimulus angle.

        Args:
            df (pd.DataFrame): Behavioral data containing 'Visual_Stim' and 'choice' columns

        Returns:
            pd.Series: Percentage of right choices for each visual stimulus angle
        """
        grouped = df.groupby('Visual_Stim')['choice'].mean() * 100
        return grouped

    def percent_right_trial_types(self, df, date):
        """
        Calculates percentage of right choices for different trial types.

        Args:
            df (pd.DataFrame): Behavioral data
            date (str): Session date

        Returns:
            dict: Contains percentages for:
                - choice_congruent: Congruent trials
                - choice_incongruent_V: Visual incongruent trials
                - choice_incongruent_A: Audio incongruent trials
        """
        trial_type_congruent = df[df['trial_type'] < 7]
        trial_type_incongruent_V = df[df['trial_type'].isin([7,8,9,10,11,12])]
        trial_type_incongruent_A = df[df['trial_type'].isin([13,14,15,16,17,18])]
        
        percent_right_choice_congruent = np.array(self.calculate_percent_right_choice(trial_type_congruent))
        percent_right_choice_incongruent_V = np.array(self.calculate_percent_right_choice(trial_type_incongruent_V))
        percent_right_choice_incongruent_A = np.array(self.calculate_percent_right_choice(trial_type_incongruent_A))

        return {
            'date': date,
            'choice_congruent': percent_right_choice_congruent, 
            'choice_incongruent_V': percent_right_choice_incongruent_V, 
            'choice_incongruent_A': percent_right_choice_incongruent_A
        }

    def logistic(self, x, alpha, beta):
        """
        Logistic function for fitting psychometric curves.

        Args:
            x (float): Input value
            alpha (float): Offset parameter
            beta (float): Slope parameter

        Returns:
            float: Logistic function output
        """
        return 1 / (1 + np.exp(-(alpha + beta * x)))

    def derivative_logistic(self, x, alpha, beta):
        """
        Calculates derivative of logistic function.

        Args:
            x (float): Input value
            alpha (float): Offset parameter
            beta (float): Slope parameter

        Returns:
            float: Derivative value
        """
        p = self.logistic(x, alpha, beta)
        return beta * p * (1 - p)

    def get_psychometric(self, orientations, probabilities):
        """
        Fits psychometric function to behavioral data.

        Args:
            orientations (np.array): Visual stimulus orientations
            probabilities (np.array): Response probabilities

        Returns:
            tuple: Contains:
                - x_values (np.array): X values for fitted curve
                - fitted_probabilities (np.array): Y values for fitted curve
                - derivatives (np.array): Slope values along curve
        """
        params, params_covariance = curve_fit(self.logistic, orientations, probabilities/100, p0=[0, 0], maxfev=500)
        x_values = np.linspace(0, 90, 300)
        fitted_probabilities = self.logistic(x_values, *params)
        derivatives = self.derivative_logistic(x_values, *params)
        return x_values, fitted_probabilities*100, derivatives

    def plot_psychometric(self, perc_right, dpi):
        """
        Creates psychometric plots for different trial types.

        Args:
            perc_right (dict): Dictionary containing performance data
            dpi (int): Resolution for plot

        Returns:
            None: Displays the plot
        """
        fig, axs = plt.subplots(1,3, figsize=(6, 2.5), dpi=dpi)

        orientations = np.array([0, 15, 25, 65, 75, 90])
        axs[0].set_title('Congruent', color='purple')
        axs[0].scatter(orientations, perc_right['choice_congruent'], color='purple')
        x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right['choice_congruent'])
        axs[0].plot(x_values, fitted_probabilities, color='purple', lw=2)

        axs[1].set_title('Incongruent-V', color='#EC008C')
        axs[1].scatter(orientations, perc_right['choice_incongruent_V'], color='#EC008C')
        x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right['choice_incongruent_V'])
        axs[1].plot(x_values, fitted_probabilities, color='#EC008C', lw=2)

        axs[2].set_title('Incongruent-A', color='#27AAE1')
        axs[2].scatter(orientations, perc_right['choice_incongruent_A'], color='#27AAE1')
        x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right['choice_incongruent_A'])
        axs[2].plot(x_values, fitted_probabilities, color='#27AAE1', lw=2)

        for x in [0,1,2]:
            axs[x].set_ylim(-5, 105)
            axs[x].axhline(y=50, color='grey', linestyle='--', linewidth=0.8)
            axs[x].axvline(x=45, color='grey', linestyle='--', linewidth=0.8)
            axs[x].set_xticks([0, 15, 30, 45, 60, 75, 90])
            axs[x].set_xticklabels([0, 15, 30, 45, 60, 75, 90], size=8)
            axs[x].set_xlabel('orientation (°)', size=10)
            
        axs[0].set_ylabel('% Right choice', size=12)
        plt.tight_layout()

    def plot_psychometric_opto(self, perc_right, perc_right_opto, type_opto, dpi):
        """
        Creates psychometric plots comparing opto and non-opto conditions.

        Args:
            perc_right (dict): Non-opto performance data
            perc_right_opto (dict): Opto performance data
            type_opto (list): Types of opto conditions to plot
            dpi (int): Resolution for plot

        Returns:
            None: Displays the plot
        """
        fig, axs = plt.subplots(1,3, figsize=(6, 2.5), dpi=dpi)

        orientations = np.array([0, 15, 25, 65, 75, 90])
        if 'No' in type_opto:
            axs[0].set_title('Congruent', color='purple')
            axs[0].scatter(orientations, perc_right['choice_congruent'], color='purple')
            x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right['choice_congruent'])
            axs[0].plot(x_values, fitted_probabilities, color='purple', lw=2, label='Non-Opto')
            
            axs[1].set_title('Incongruent-V', color='#EC008C')
            axs[1].scatter(orientations, perc_right['choice_incongruent_V'], color='#EC008C')
            x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right['choice_incongruent_V'])
            axs[1].plot(x_values, fitted_probabilities, color='#EC008C', lw=2)

            axs[2].set_title('Incongruent-A', color='#27AAE1')
            axs[2].scatter(orientations, perc_right['choice_incongruent_A'], color='#27AAE1')
            x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right['choice_incongruent_A'])
            axs[2].plot(x_values, fitted_probabilities, color='#27AAE1', lw=2)

        if 0 in type_opto:
            axs[0].scatter(orientations, perc_right_opto['choice_congruent'], color='purple', alpha=0.5, marker='*')
            x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right_opto['choice_congruent'])
            axs[0].plot(x_values, fitted_probabilities, color='purple', lw=2, alpha=0.5, label='Opto')

        if 1 in type_opto:
            axs[1].scatter(orientations, perc_right_opto['choice_incongruent_V'], color='#EC008C', alpha=0.5, marker='*')
            x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right_opto['choice_incongruent_V'])
            axs[1].plot(x_values, fitted_probabilities, color='#EC008C', lw=2, alpha=0.5)

        if 2 in type_opto:
            axs[2].scatter(orientations, perc_right_opto['choice_incongruent_A'], color='#27AAE1', alpha=0.5, marker='*')
            x_values, fitted_probabilities, derivatives = self.get_psychometric(orientations, perc_right_opto['choice_incongruent_A'])
            axs[2].plot(x_values, fitted_probabilities, color='#27AAE1', lw=2, alpha=0.5)

        for x in [0,1,2]:
            axs[x].set_ylim(-5, 105)
            axs[x].axhline(y=50, color='grey', linestyle='--', linewidth=0.8)
            axs[x].axvline(x=45, color='grey', linestyle='--', linewidth=0.8)
            axs[x].set_xticks([0, 15, 30, 45, 60, 75, 90])
            axs[x].set_xticklabels([0, 15, 30, 45, 60, 75, 90], size=8)
            axs[x].set_xlabel('orientation (°)', size=10)
            
        axs[0].set_ylabel('% Right choice', size=12)
        axs[0].legend(loc=4, frameon=False, fontsize='xx-small')
        plt.tight_layout()

class behavioral_models:
    """
    Class for running behavioral models, specifically GLM, on behavioral data.
    """

    def __init__(self, df):
        """
        Initializes the behavioral_models class with behavioral data.

        Args:
            df (pd.DataFrame): DataFrame containing behavioral data, expected to be the output from get_data_multiple_sessions.
        """
        self.df = df

    def get_beta_weights(self, results):
        """
        Extracts standardized beta weights from GLM results.

        Args:
            results (statsmodels.genmod.generalized_linear_model.GLMResultsWrapper): GLM results object

        Returns:
            pd.Series: Series containing the standardized beta weights for each predictor
        """
        return pd.Series(results.params, index=results.model.exog_names)

    def run_glm(self, task_type='psych'):
        """
        Runs a Generalized Linear Model (GLM) on the behavioral data with z-scored inputs.
        
        Args:
            task_type (str): The type of task ('aud', 'aud_psych', '2loc', or 'psych'). Default is 'psych'.
            
        Returns:
            dict: Dictionary of beta weights for each context
        """
        beta_weights_dict = {}

        if task_type == 'aud' or task_type == 'aud_psych':
            # Filter out rows where 'Audio_Stim' is NaN
            df_filtered = self.df[self.df['Audio_Stim'].notna()].copy()
            
            # Define predictors and response variable
            X = df_filtered[['Audio_Stim']]
            y = df_filtered['choice']

            # Add previous choice and previous outcome as predictors
            X['prev_choice'] = y.shift(1)
            X['prev_outcome'] = df_filtered['outcome'].shift(1)

            # Drop rows with NaN values resulting from the shift
            mask = X.notna().all(axis=1)
            X = X[mask]
            y = y[mask]

            # Check if we have enough data points
            if len(X) < 10:  # Minimum number of trials needed
                print("Warning: Not enough valid trials for GLM")
                return None

            try:
                # Z-score the predictors, handling potential inf/nan values
                X = pd.DataFrame(zscore(X, nan_policy='omit'), columns=X.columns, index=X.index)
                
                # Replace any remaining inf values with large finite numbers
                X = X.replace([np.inf, -np.inf], [1e10, -1e10])
                
                # Add a constant to the model (for the intercept)
                X = sm.add_constant(X)

                # Create and fit the GLM model
                model = sm.GLM(y, X, family=sm.families.Binomial())
                results = model.fit()
                beta_weights_dict['all'] = self.get_beta_weights(results)
            except Exception as e:
                print(f"GLM fitting failed: {str(e)}")
                return None

        elif task_type == '2loc' or task_type == 'psych':
            # Filter out rows where 'Audio_Stim' or 'Visual_Stim' is NaN
            df_filtered = self.df[self.df['Audio_Stim'].notna() & 
                                self.df['Visual_Stim'].notna()].copy()

            # Get unique context values
            contexts = df_filtered['context'].unique()

            for context in contexts:
                # Filter data for the current context
                df_context = df_filtered[df_filtered['context'] == context]

                # Define predictors and response variable
                X = df_context[['Audio_Stim', 'Visual_Stim']]
                y = df_context['choice']

                # Add previous choice and previous outcome as predictors
                X['prev_choice'] = y.shift(1)
                X['prev_outcome'] = df_context['outcome'].shift(1)

                # Drop rows with NaN values
                mask = X.notna().all(axis=1)
                X = X[mask]
                y = y[mask]

                # Check if we have enough data points
                if len(X) < 10:  # Minimum number of trials needed
                    print(f"Warning: Not enough valid trials for context {context}")
                    beta_weights_dict[context] = None
                    continue

                try:
                    # Z-score the predictors, handling potential inf/nan values
                    X = pd.DataFrame(zscore(X, nan_policy='omit'), columns=X.columns, index=X.index)
                    
                    # Replace any remaining inf values with large finite numbers
                    X = X.replace([np.inf, -np.inf], [1e10, -1e10])
                    
                    # Add a constant to the model (for the intercept)
                    X = sm.add_constant(X)

                    # Create and fit the GLM model
                    model = sm.GLM(y, X, family=sm.families.Binomial())
                    results = model.fit()
                    beta_weights_dict[context] = self.get_beta_weights(results)
                except Exception as e:
                    print(f"GLM fitting failed for context {context}: {str(e)}")
                    beta_weights_dict[context] = None
                    continue

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        return beta_weights_dict

    def run_glm_with_bias(self, task_type='psych'):
        """
        Runs GLM including a bias term (constant) and tracks its magnitude.
        Similar to run_glm but explicitly separates and tracks the bias parameter.
        
        Args:
            task_type (str): The type of task ('aud', 'aud_psych', '2loc', or 'psych'). Default is 'psych'.
            
        Returns:
            tuple: (beta_weights_dict, bias_dict) containing weights and bias terms for each context
        """
        beta_weights_dict = {}
        bias_dict = {}

        if task_type in ['aud', 'aud_psych']:
            # Filter out rows where 'Audio_Stim' is NaN
            df_filtered = self.df[self.df['Audio_Stim'].notna()].copy()
            
            # Define predictors and response variable
            X = pd.DataFrame({'Audio_Stim': df_filtered['Audio_Stim']})
            y = df_filtered['choice']

            # Add previous choice and previous outcome as predictors
            X['prev_choice'] = y.shift(1)
            X['prev_outcome'] = df_filtered['outcome'].shift(1)

            # Drop rows with NaN values
            mask = X.notna().all(axis=1)
            X = X[mask]
            y = y[mask]

            # Check if we have enough data points
            if len(X) < 10:
                print("Warning: Not enough valid trials for GLM")
                return None, None

            try:
                # Z-score the predictors, handling potential inf/nan values
                X = pd.DataFrame(zscore(X, nan_policy='omit'), columns=X.columns, index=X.index)
                
                # Replace any remaining inf values with large finite numbers
                X = X.replace([np.inf, -np.inf], [1e10, -1e10])
                
                # Add bias term explicitly
                X.insert(0, 'bias', 1.0)

                # Create and fit the GLM model
                model = sm.GLM(y, X, family=sm.families.Binomial())
                results = model.fit()
                
                beta_weights_dict['all'] = pd.Series(results.params[1:], index=X.columns[1:])  # Exclude bias
                bias_dict['all'] = results.params[0]  # Store bias separately
                
            except Exception as e:
                print(f"GLM fitting failed: {str(e)}")
                return None, None

        elif task_type in ['2loc', 'psych']:
            # Filter out rows where 'Audio_Stim' or 'Visual_Stim' is NaN
            df_filtered = self.df[self.df['Audio_Stim'].notna() & 
                                self.df['Visual_Stim'].notna()].copy()

            # Get unique context values
            contexts = df_filtered['context'].unique()

            for context in contexts:
                # Filter data for the current context
                df_context = df_filtered[df_filtered['context'] == context]

                # Define predictors and response variable
                X = df_context[['Audio_Stim', 'Visual_Stim']]
                y = df_context['choice']

                # Add previous choice and previous outcome as predictors
                X['prev_choice'] = y.shift(1)
                X['prev_outcome'] = df_context['outcome'].shift(1)

                # Drop rows with NaN values
                mask = X.notna().all(axis=1)
                X = X[mask]
                y = y[mask]

                # Check if we have enough data points
                if len(X) < 10:
                    print(f"Warning: Not enough valid trials for context {context}")
                    beta_weights_dict[context] = None
                    bias_dict[context] = None
                    continue

                try:
                    # Z-score the predictors, handling potential inf/nan values
                    X = pd.DataFrame(zscore(X, nan_policy='omit'), columns=X.columns, index=X.index)
                    
                    # Replace any remaining inf values with large finite numbers
                    X = X.replace([np.inf, -np.inf], [1e10, -1e10])
                    
                    # Add bias term explicitly
                    X.insert(0, 'bias', 1.0)

                    # Create and fit the GLM model
                    model = sm.GLM(y, X, family=sm.families.Binomial())
                    results = model.fit()
                    
                    beta_weights_dict[context] = pd.Series(results.params[1:], index=X.columns[1:])  # Exclude bias
                    bias_dict[context] = results.params[0]  # Store bias separately
                    
                except Exception as e:
                    print(f"GLM fitting failed for context {context}: {str(e)}")
                    beta_weights_dict[context] = None
                    bias_dict[context] = None
                    continue

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        return beta_weights_dict, bias_dict

    def fit_learning_curves(self, sessions_data, task_type='psych'):
        """
        Fits exponential learning curves to sensory weights over sessions.
        
        Args:
            sessions_data (list): List of DataFrames, one per session
            task_type (str): Type of task
            
        Returns:
            dict: Contains fitted parameters and curves for each modality and context
        """
        def exp_func(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c
        
        # Get weights for each session
        session_weights = []
        for session_df in sessions_data:
            self.df = session_df  # Temporarily set df for GLM
            weights, _ = self.run_glm_with_bias(task_type)
            if weights is not None:  # Only append if GLM fitting succeeded
                session_weights.append(weights)
        
        if not session_weights:  # If no successful fits
            print("No successful GLM fits for learning curve analysis")
            return {}
        
        # Organize weights by modality and context
        results = {}
        
        # Handle different task types
        if task_type in ['aud', 'aud_psych']:
            # For auditory tasks, we only have one context ('all')
            if 'all' in session_weights[0]:
                audio_weights = [w['all']['Audio_Stim'] for w in session_weights if w['all'] is not None]
                
                if audio_weights:  # If we have weights to fit
                    try:
                        x = np.arange(len(audio_weights))
                        popt_audio, _ = curve_fit(exp_func, x, audio_weights)
                        results['audio'] = {
                            'parameters': {
                                'amplitude': popt_audio[0],
                                'rate': popt_audio[1],
                                'offset': popt_audio[2]
                            },
                            'fitted_values': exp_func(x, *popt_audio).tolist(),
                            'raw_weights': audio_weights,
                            'sessions': list(range(len(audio_weights)))
                        }
                    except Exception as e:
                        print(f"Failed to fit audio learning curve: {str(e)}")
                        results['audio'] = None
        
        elif task_type in ['2loc', 'psych']:
            # Get all unique contexts from all sessions
            all_contexts = set()
            for weights in session_weights:
                all_contexts.update(weights.keys())
            
            # For each context
            for context in all_contexts:
                context_results = {}
                
                # Get audio weights for this context
                audio_weights = []
                visual_weights = []
                
                for w in session_weights:
                    if context in w and w[context] is not None:
                        if 'Audio_Stim' in w[context]:
                            audio_weights.append(w[context]['Audio_Stim'])
                        if 'Visual_Stim' in w[context]:
                            visual_weights.append(w[context]['Visual_Stim'])
                
                if audio_weights:  # Fit audio weights
                    try:
                        x = np.arange(len(audio_weights))
                        popt_audio, _ = curve_fit(exp_func, x, audio_weights)
                        context_results['audio'] = {
                            'parameters': {
                                'amplitude': popt_audio[0],
                                'rate': popt_audio[1],
                                'offset': popt_audio[2]
                            },
                            'fitted_values': exp_func(x, *popt_audio).tolist(),
                            'raw_weights': audio_weights,
                            'sessions': list(range(len(audio_weights)))
                        }
                    except Exception as e:
                        print(f"Failed to fit audio learning curve for context {context}: {str(e)}")
                        context_results['audio'] = None
                
                if visual_weights:  # Fit visual weights
                    try:
                        x = np.arange(len(visual_weights))
                        popt_visual, _ = curve_fit(exp_func, x, visual_weights)
                        context_results['visual'] = {
                            'parameters': {
                                'amplitude': popt_visual[0],
                                'rate': popt_visual[1],
                                'offset': popt_visual[2]
                            },
                            'fitted_values': exp_func(x, *popt_visual).tolist(),
                            'raw_weights': visual_weights,
                            'sessions': list(range(len(visual_weights)))
                        }
                    except Exception as e:
                        print(f"Failed to fit visual learning curve for context {context}: {str(e)}")
                        context_results['visual'] = None
                
                if audio_weights or visual_weights:  # Only add context if we have data
                    results[f'context_{context}'] = context_results
        
        return results

    def print_learning_curve_results(self, learning_curves):
        """
        Helper method to print learning curve results in a readable format.
        
        Args:
            learning_curves (dict): Results from fit_learning_curves
        """
        if not learning_curves:
            print("No learning curve results available")
            return
        
        for context, modalities in learning_curves.items():
            print(f"\n{context.replace('_', ' ').title()}:")
            
            if isinstance(modalities, dict):  # Check if it's a context with multiple modalities
                for modality, results in modalities.items():
                    if results is not None:
                        print(f"\n  {modality.capitalize()}:")
                        print(f"    Parameters:")
                        print(f"      Amplitude: {results['parameters']['amplitude']:.3f}")
                        print(f"      Learning rate: {results['parameters']['rate']:.3f}")
                        print(f"      Offset: {results['parameters']['offset']:.3f}")
                        print(f"    Number of sessions: {len(results['sessions'])}")
                        print(f"    Raw weights range: {min(results['raw_weights']):.3f} to {max(results['raw_weights']):.3f}")
            else:  # Single modality result
                if modalities is not None:
                    print(f"  Parameters:")
                    print(f"    Amplitude: {modalities['parameters']['amplitude']:.3f}")
                    print(f"    Learning rate: {modalities['parameters']['rate']:.3f}")
                    print(f"    Offset: {modalities['parameters']['offset']:.3f}")
                    print(f"  Number of sessions: {len(modalities['sessions'])}")
                    print(f"  Raw weights range: {min(modalities['raw_weights']):.3f} to {max(modalities['raw_weights']):.3f}")

    def compute_choice_bias(self, rolling_window=100):
        """
        Computes running choice bias (preference for left/right) over trials.
        
        Args:
            rolling_window (int): Number of trials for rolling average
            
        Returns:
            pd.Series: Rolling choice bias
        """
        # Convert choices to -1 (left) and 1 (right)
        choices = self.df['choice'].map({0: -1, 1: 1})
        
        # Compute rolling mean (positive = right bias, negative = left bias)
        bias = choices.rolling(window=rolling_window, center=True).mean()
        
        return bias

    def compute_psychometric_params(self, task_type='psych'):
        """
        Computes parameters of psychometric function (slope, bias) for each context.
        
        Returns:
            dict: Contains slope and bias parameters for each context
        """
        params_dict = {}
        
        if task_type in ['2loc', 'psych']:
            for context in self.df['context'].unique():
                df_context = self.df[self.df['context'] == context]
                
                # Group by visual stimulus and get mean right choices
                grouped = df_context.groupby('Visual_Stim')['choice'].mean()
                
                # Fit logistic function
                x = np.array(grouped.index)
                y = grouped.values
                
                try:
                    # Fit logistic function (similar to psychometrics class)
                    popt, _ = curve_fit(lambda x, bias, slope: 
                                      1 / (1 + np.exp(-(bias + slope * x))),
                                      x, y, p0=[0, 0.1])
                    
                    params_dict[context] = {
                        'bias': popt[0],
                        'slope': popt[1]
                    }
                except:
                    params_dict[context] = None
                    
        return params_dict
