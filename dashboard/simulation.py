"""
simulation.py
Airport Operations and Live Feed Simulator.
Simulates real-time radar stream updates, allowing the user to advance time,
play continuously, or inject severe operational weather/traffic disruptions.
"""

import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hourly_airport_traffic.csv")

class TrafficSimulator:
    def __init__(self, airport: str = "HYD"):
        self.airport = airport
        self.df = None
        self.cursor = 24 # Start at day 2 so we have plenty of lookback history
        self.load_data()
        
    def load_data(self):
        if os.path.exists(DATA_PATH):
            df_all = pd.read_csv(DATA_PATH)
            self.df = df_all[df_all["airport"] == self.airport].sort_values("timestamp").reset_index(drop=True)
        else:
            self.df = pd.DataFrame()
            
    def set_airport(self, airport: str):
        self.airport = airport
        self.cursor = 24
        self.load_data()
        
    def get_max_steps(self) -> int:
        return len(self.df) if self.df is not None else 0
        
    def step_forward(self, steps: int = 1):
        if self.df is not None and len(self.df) > 0:
            self.cursor = min(len(self.df) - 1, self.cursor + steps)
            
    def reset(self):
        self.cursor = 24
        
    def get_recent_history(
        self, 
        lookback: int = 6, 
        weather_injection: float = 0.0, 
        surge_injection: float = 0.0,
        runway_reduction: float = 0.0
    ) -> pd.DataFrame:
        """
        Retrieves the past lookback hours up to the current cursor.
        Applies live stress injection parameters if set by user.
        """
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()
            
        start_idx = max(0, self.cursor - lookback + 1)
        sub_df = self.df.iloc[start_idx : self.cursor + 1].copy()
        
        # Apply stress injections to the most recent hour (current radar snapshot)
        if weather_injection > 0 or surge_injection > 0 or runway_reduction > 0:
            last_idx = sub_df.index[-1]
            
            if surge_injection > 0:
                surge_mult = 1.0 + (surge_injection / 100.0)
                sub_df.loc[last_idx, "arriving_flights"] = int(sub_df.loc[last_idx, "arriving_flights"] * surge_mult)
                sub_df.loc[last_idx, "departing_flights"] = int(sub_df.loc[last_idx, "departing_flights"] * surge_mult)
                sub_df.loc[last_idx, "total_flights"] = (
                    sub_df.loc[last_idx, "arriving_flights"] + sub_df.loc[last_idx, "departing_flights"]
                )
                
            if weather_injection > 0:
                sub_df.loc[last_idx, "weather_delay_factor"] += weather_injection
                sub_df.loc[last_idx, "avg_departure_delay"] += (weather_injection * 1.5)
                sub_df.loc[last_idx, "avg_arrival_delay"] += (weather_injection * 1.2)
                
            capacity = sub_df.loc[last_idx, "airport_capacity"]
            if runway_reduction > 0:
                capacity = max(10, int(capacity * (1.0 - runway_reduction / 100.0)))
                sub_df.loc[last_idx, "airport_capacity"] = capacity
                
            total = sub_df.loc[last_idx, "total_flights"]
            sub_df.loc[last_idx, "runway_utilization"] = min(100.0, round((total / capacity) * 100.0, 1))
            
            # Recalculate congestion level
            util = sub_df.loc[last_idx, "runway_utilization"]
            dep_delay = sub_df.loc[last_idx, "avg_departure_delay"]
            if util >= 85.0 or dep_delay >= 32.0:
                sub_df.loc[last_idx, "congestion_level"] = 2
            elif util >= 65.0 or dep_delay >= 16.0:
                sub_df.loc[last_idx, "congestion_level"] = 1
            else:
                sub_df.loc[last_idx, "congestion_level"] = 0

        return sub_df
