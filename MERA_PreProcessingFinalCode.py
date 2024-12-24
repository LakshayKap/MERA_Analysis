# Weather Data Preprocessing Pipeline
# Author: Data Science Team
# Last Updated: December 2024
# Description: Preprocesses weather station data from Dublin Airport for analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class WeatherDataPreprocessor:
    """A class to handle preprocessing of weather station data."""
    
    def __init__(self, file_path):
        """
        Initialize the preprocessor with input file path.
        
        Args:
            file_path (str): Path to the input CSV file
        """
        self.file_path = file_path
        self.station_info = {
            'latitude': 53.428,
            'longitude': -6.241,
            'station_name': 'Dublin Airport'
        }
        
    def load_data(self):
        """Load and initialize the dataset with basic cleaning."""
        # Read and clean header
        self.df = pd.read_csv(self.file_path, delimiter=',')
        self.df.columns = self.df.iloc[0]
        self.df = self.df[1:]
        self.df.reset_index(drop=True, inplace=True)
        
        # Add station information
        self.df['Latitude'] = self.station_info['latitude']
        self.df['Longitude'] = self.station_info['longitude']
        self.df['Station Name'] = self.station_info['station_name']
        
    def convert_datatypes(self):
        """Convert columns to appropriate data types."""
        # Convert date and categorical columns
        self.df["date"] = pd.to_datetime(self.df["date"])
        
        # Convert numeric columns
        numeric_columns = ["maxtp", "mintp", "rain", "wdsp", "sun", "soil", 
                         "smd_wd", "smd_md", "smd_pd", "hm", "hg", "ddhm", 
                         "gmin", "dos", "evap"]
        self.df[numeric_columns] = self.df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        
        # Drop redundant columns
        self.df = self.df.drop(columns="ind")
        
    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        # Fill missing values with mean for specific columns
        mean_fill_columns = ['hm', 'ddhm', 'hg', 'soil']
        for col in mean_fill_columns:
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        # Drop rows with missing soil moisture data
        self.df.dropna(subset=['smd_wd', 'smd_md', 'smd_pd'], inplace=True, axis=0)
        
    def create_date_features(self):
        """Extract date components and create seasonal features."""
        # Extract year, month, day
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        
        # Create season column
        season_mapping = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 
            4: 'Spring', 5: 'Spring', 6: 'Summer',
            7: 'Summer', 8: 'Summer', 9: 'Autumn', 
            10: 'Autumn', 11: 'Autumn', 12: 'Winter'
        }
        self.df['season'] = self.df['date'].dt.month.map(season_mapping)
        
    def handle_outliers(self):
        """Handle outliers using IQR method."""
        # Calculate bounds for rain data
        Q1 = self.df['rain'].quantile(0.25)
        Q3 = self.df['rain'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Clip outliers
        columns_to_clip = ['rain', 'smd_wd', 'smd_md', 'smd_pd']
        for col in columns_to_clip:
            self.df[col] = self.df[col].clip(lower_bound, upper_bound)
            
    def create_derived_features(self):
        """Create derived features and weather categories."""
        # Temperature range
        self.df['temp_range'] = self.df['max_temp'] - self.df['min_temp']
        
        # Combined soil moisture
        self.df['smd_combined'] = self.df[['smd_wd', 'smd_md', 'smd_pd']].mean(axis=1)
        
        # Wind direction categories
        self.df['wind_category'] = self.df['dir_10minute_wind'].apply(self._categorize_wind_direction)
        
        # Create weather condition flags
        self.df['extreme_rain'] = (self.df['rain'] > 2.15).astype(int)
        self.df['heatwave'] = (self.df['max_temp'] > 3.21).astype(int)
        self.df['high_wind'] = (self.df['max_gust'] > 5.59).astype(int)
        
        # Combined extreme weather indicator
        self.df['extreme_weather'] = (
            (self.df['extreme_rain'] == 1) | 
            (self.df['heatwave'] == 1) | 
            (self.df['high_wind'] == 1)
        ).astype(int)
        
    def _categorize_wind_direction(self, deg):
        """
        Categorize wind direction into compass directions.
        
        Args:
            deg (float): Wind direction in degrees
            
        Returns:
            str: Wind direction category
        """
        if pd.isna(deg):
            return 'Invalid'
        
        deg = float(deg)
        if deg >= 337.5 or deg < 22.5:
            return 'N'
        elif 22.5 <= deg < 67.5:
            return 'NE'
        elif 67.5 <= deg < 112.5:
            return 'E'
        elif 112.5 <= deg < 157.5:
            return 'SE'
        elif 157.5 <= deg < 202.5:
            return 'S'
        elif 202.5 <= deg < 247.5:
            return 'SW'
        elif 247.5 <= deg < 292.5:
            return 'W'
        else:
            return 'NW'
            
    def standardize_features(self):
        """Standardize numeric features."""
        numeric_columns = ['max_temp', 'min_temp', 'wind_speed', 'max_10minute_wind',
                         'max_gust', 'dir_10minute_wind', 'min_grass', 'dos', 'evap']
        scaler = StandardScaler()
        self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
        
    def validate_data(self):
        """Validate dataset completeness and quality."""
        # Check for missing dates
        complete_dates = pd.date_range(start=self.df['date'].min(), 
                                     end=self.df['date'].max(), 
                                     freq='D')
        missing_dates = set(complete_dates) - set(self.df['date'])
        if missing_dates:
            print(f"Warning: Found {len(missing_dates)} missing dates in the dataset")
            
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Warning: Found {duplicates} duplicate rows")
            
    def process_data(self):
        """Execute the complete preprocessing pipeline."""
        print("Starting data preprocessing...")
        self.load_data()
        self.convert_datatypes()
        self.handle_missing_values()
        self.create_date_features()
        self.handle_outliers()
        self.create_derived_features()
        self.standardize_features()
        self.validate_data()
        print("Preprocessing complete!")
        
    def save_processed_data(self, output_path):
        """
        Save the processed dataset to CSV.
        
        Args:
            output_path (str): Path for the output CSV file
        """
        self.df = self.df.round(2)  # Round to 2 decimal places
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

def main():
    """Main execution function."""
    # Initialize and run preprocessing
    preprocessor = WeatherDataPreprocessor("dly532.csv")
    preprocessor.process_data()
    preprocessor.save_processed_data("Mera_CleanData.csv")
    
    print("\nDataset Summary:")
    print(f"Total rows: {len(preprocessor.df)}")
    print(f"Total columns: {len(preprocessor.df.columns)}")
    print(f"Date range: {preprocessor.df['date'].min()} to {preprocessor.df['date'].max()}")

if __name__ == "__main__":
    main()
