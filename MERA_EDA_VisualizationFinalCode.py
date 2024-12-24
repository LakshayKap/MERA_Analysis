# Weather Data Exploratory Analysis and Visualization
# Author: Data Science Team
# Last Updated: December 2024
# Description: Creates comprehensive visualizations for weather data analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WeatherDataVisualizer:
    """A class to handle exploratory data analysis and visualization of weather data."""
    
    def __init__(self, data_path):
        """
        Initialize the visualizer with input file path.
        
        Args:
            data_path (str): Path to the processed CSV file
        """
        self.data_path = data_path
        self.figures = {}
        self.load_data()
        
    def load_data(self):
        """Load and prepare the dataset for visualization."""
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print("Dataset loaded successfully!")
        print("\nDataset Info:")
        print(self.df.info())
        
    def create_target_distribution(self):
        """Create distribution plot of extreme weather events."""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x="extreme_weather", palette="viridis")
        plt.title("Distribution of Extreme Weather Events", fontsize=12)
        plt.xlabel("Extreme Weather (0: No, 1: Yes)", fontsize=10)
        plt.ylabel("Count", fontsize=10)
        self.figures["extreme_weather_distribution"] = plt.gcf()
        plt.close()
        
    def create_correlation_heatmap(self):
        """Create correlation heatmap for numerical features."""
        plt.figure(figsize=(12, 10))
        numerical_features = self.df.select_dtypes(include=["float64", "int64"])
        corr_matrix = numerical_features.corr()
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap of Weather Features", fontsize=12)
        self.figures["correlation_heatmap"] = plt.gcf()
        plt.close()
        
    def create_weather_parameter_boxplots(self):
        """Create boxplots for key weather parameters by extreme weather status."""
        parameters = {
            'max_temp': 'Temperature',
            'rain': 'Precipitation',
            'wind_speed': 'Wind Speed'
        }
        
        for param, title in parameters.items():
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=self.df, x="extreme_weather", y=param, palette="Set2")
            plt.title(f"{title} vs Extreme Weather", fontsize=12)
            plt.xlabel("Extreme Weather (0: No, 1: Yes)", fontsize=10)
            plt.ylabel(title, fontsize=10)
            self.figures[f"{param}_boxplot"] = plt.gcf()
            plt.close()
            
    def create_feature_pairplot(self):
        """Create pairplot for selected weather features."""
        selected_features = ["rain", "pressure_cbl", "wind_speed", 
                           "temp_range", "extreme_weather"]
        pairplot = sns.pairplot(self.df[selected_features], 
                               hue="extreme_weather", 
                               palette="husl")
        self.figures["feature_pairplot"] = pairplot.fig
        plt.close()
        
    def create_time_series_plots(self):
        """Create time series plots for key weather parameters."""
        parameters = {
            'max_temp': 'Temperature',
            'rain': 'Precipitation',
            'wind_speed': 'Wind Speed'
        }
        
        for param, title in parameters.items():
            plt.figure(figsize=(12, 6))
            plt.plot(self.df['date'], self.df[param])
            plt.title(f"{title} Time Series", fontsize=12)
            plt.xlabel("Date", fontsize=10)
            plt.ylabel(title, fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.figures[f"{param}_timeseries"] = plt.gcf()
            plt.close()
            
    def create_seasonal_analysis(self):
        """Create seasonal analysis plots."""
        plt.figure(figsize=(10, 6))
        seasonal_extreme = self.df.groupby('season')['extreme_weather'].mean()
        seasonal_extreme.plot(kind='bar')
        plt.title("Proportion of Extreme Weather Events by Season", fontsize=12)
        plt.xlabel("Season", fontsize=10)
        plt.ylabel("Proportion of Extreme Events", fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.figures["seasonal_analysis"] = plt.gcf()
        plt.close()
        
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("Generating visualizations...")
        self.create_target_distribution()
        self.create_correlation_heatmap()
        self.create_weather_parameter_boxplots()
        self.create_feature_pairplot()
        self.create_time_series_plots()
        self.create_seasonal_analysis()
        print("All visualizations generated successfully!")
        
    def save_plots(self, output_dir):
        """
        Save all generated plots to specified directory.
        
        Args:
            output_dir (str): Directory path to save the plots
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for name, fig in self.figures.items():
            output_path = os.path.join(output_dir, f"{name}.png")
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
        
        print(f"All plots saved to {output_dir}")
        
    def display_summary_statistics(self):
        """Display summary statistics of the dataset."""
        print("\nSummary Statistics:")
        print(self.df.describe())
        
        print("\nExtreme Weather Event Summary:")
        print(self.df['extreme_weather'].value_counts(normalize=True))

def main():
    """Main execution function."""
    # Initialize visualizer
    visualizer = WeatherDataVisualizer("Mera_CleanData.csv")
    
    # Generate and save all plots
    visualizer.generate_all_plots()
    visualizer.save_plots("weather_visualizations")
    
    # Display summary statistics
    visualizer.display_summary_statistics()

if __name__ == "__main__":
    main()
