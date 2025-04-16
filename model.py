import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import csv

import mesa
import random

from mesa.space import MultiGrid, PropertyLayer
from datetime import datetime, timedelta
from mesa.datacollection import DataCollector



class Cow(mesa.Agent):
    """
    This class represents an individual cow in the simulation.
    It models the cow's growth, feed intake, and energy requirements
    based on the Cornell Value Discovery System (CVDS).
    """
    def __init__(self, model, cattle_info):
        super().__init__(model)

        # Basic attributes from input data
        self.hip_height = float(cattle_info["HipHeight"])  # Hip height in inches
        self.age = float(cattle_info["AgeHipHeight"]) * 30  # Age in days
        self.cattle_type = cattle_info["Beef"]  # Type of cattle (Beef/Dairy)
        self.gender = cattle_info["Sex"]  # Gender: Bull (B), Heifer (H), or Steer (S)
        self.full_body_weight = float(cattle_info["iBW"])  # Initial body weight (BW)

        # Determine shrunk body weight (SBW) based on CVDS standards
        self.Shrunk_Body_Weight = (
            self.full_body_weight if cattle_info["IsiBWShrunk"] == "TRUE" else 0.94 * self.full_body_weight  # 94% of full body weight if not shrunk
        )

        self.Adjusted_Final_Body_Weight = 0.0  # Placeholder for calculated final BW
        self.usingImplants = cattle_info["Implants"] == "TRUE"  # Growth implant usage
        self.holsteinBreeding = cattle_info["Holstein"] == "TRUE"  # Holstein cattle flag
        self.dmi = 0.0  # Dry Matter Intake
        self.Body_Condition_Score = int(cattle_info["BCS"])  # Body Condition Score (1-9)
        self.Feed_Required = 0.0  # Feed required for maintenance and growth
        self.Retained_Energy = 0.0  # Energy retained for growth

        # Calculate Frame Score & Adjusted Final BW using CVDS equations
        if self.gender == "B":  # Bull
            self.frame_score = -11.5480 + 0.4878 * self.hip_height - 0.0289 * self.age + 0.00001947 * self.age**2 + 0.0000334 * self.hip_height * self.age
            self.Adjusted_Final_Body_Weight = 33.35 * self.frame_score + 366.52
        else:  # Heifers and Steers
            self.frame_score = -11.7086 + 0.4723 * self.hip_height - 0.0239 * self.age + 0.00001460 * self.age**2 + 0.0000759 * self.hip_height * self.age
            self.Adjusted_Final_Body_Weight = 26.70 * self.frame_score + 293.2

        if self.gender == "H":
            self.Equivalent_Shrunk_Body_Weight = (478 * self.Shrunk_Body_Weight) / self.Adjusted_Final_Body_Weight
        else: # For Finishing/Growing
            self.Equivalent_Shrunk_Body_Weight = (478 * self.Shrunk_Body_Weight) / self.Adjusted_Final_Body_Weight
        if self.Equivalent_Shrunk_Body_Weight >= 478:
            self.Equivalent_Shrunk_Body_Weight = 478

        # Initial values for fat, energy balance, and carcass characteristics
        self.Fat = 0.0
        self.YG = 5  # Yield Grade assumption for quality meat grading
        self.Shrunken_Weight_Gain = 0.0
        self.Empty_Weight_Gain = 0.0
        self.Fat_In_EWG = 0.0
        self.Expected_EBF = 4.749 + 7.861 * self.YG - 8.006 * (1.052 - 0.0317 * self.YG + 0.0051 * self.YG)**0.5
        self.Empty_Body_Weight = 0.891 * self.Shrunk_Body_Weight  # 89.1% of SBW
        self.Empty_Body_Fat = 0.244 * self.Empty_Body_Weight - 15.4135
        # self.Empty_Body_Fat= 17.11
        self.Equivalent_Carcass_Weight = (0.891 * self.Equivalent_Shrunk_Body_Weight - 30.26) / 1.36
        self.Carcass_Weight_Percent = self.Equivalent_Carcass_Weight / self.Equivalent_Shrunk_Body_Weight
        self.Carcass_Weight = self.Carcass_Weight_Percent * self.Shrunk_Body_Weight
        self.Carcass_Weight_Gain = 0.0

    def feed(self):
        """
        Simulates the cow's feed intake, energy retention, and body weight gain.
        """
        pen_id, _ = self.pos  # Get pen location from model
        NEm_diet = self.model.pens.properties["NEm_diet"].data[pen_id, 0]  # Net Energy for Maintenance
        NEg_diet = self.model.pens.properties["NEg_diet"].data[pen_id, 0]  # Net Energy for Gain



        # Calculate Dry Matter Intake (DMI)
        self.dmi = self.dry_matter_intake(NEm_diet) *2 + 5
        # Compute Energy for Maintenance and Feed Required
        self.NEm_req = self.net_energy_maintainence()
        self.Feed_Required = self.NEm_req /(NEm_diet * 1.12)



        # Calculate Retained Energy (RE) for growth
        self.Retained_Energy = (self.dmi - self.Feed_Required) * NEg_diet
        self.Retained_Energy = 0 if self.Retained_Energy < 0 else self.Retained_Energy

        # Update Shrunk BW and Carcass Weight
        self.Shrunken_Weight_Gain = 13.91 * (self.Equivalent_Shrunk_Body_Weight** (- 0.6837)) * self.Retained_Energy**0.9116
        self.Shrunk_Body_Weight += self.Shrunken_Weight_Gain

        self.Empty_Weight_Gain = 0.956 * self.Shrunken_Weight_Gain
        self.Fat_In_EWG = 0.122 * self.Retained_Energy / self.Empty_Weight_Gain - 0.146
        self.Fat += self.Fat_In_EWG * self.Empty_Weight_Gain
        self.Empty_Body_Weight = 0.891 * self.Shrunk_Body_Weight
        self.Empty_Body_Fat = self.Fat * 100 / self.Empty_Body_Weight

        self.Equivalent_Carcass_Weight = (0.891 * self.Equivalent_Shrunk_Body_Weight - 30.26) / 1.36
        self.Carcass_Weight_Percent = self.Equivalent_Carcass_Weight / self.Equivalent_Shrunk_Body_Weight
        new_Carcass_Weight = self.Carcass_Weight_Percent * self.Shrunk_Body_Weight

        # Recalculate carcass weight based on new BW
        self.Carcass_Weight_Gain = new_Carcass_Weight - self.Carcass_Weight
        self.Carcass_Weight = new_Carcass_Weight

        if self.gender == "B" :
            self.frame_score = -11.5480 + 0.4878 * self.hip_height - 0.0289 * self.age + 0.00001947 * self.age**2 + 0.0000334 * self.hip_height*self.age
            self.Adjusted_Final_Body_Weight = 33.35 * self.frame_score + 366.52
        else:
            self.frame_score = -11.7086 + 0.4723 * self.hip_height - 0.0239 * self.age + 0.00001460 * self.age**2 + 0.0000759 * self.hip_height*self.age
            self.Adjusted_Final_Body_Weight = 26.70 * self.frame_score + 293.2
        if self.gender == "H":
            self.Equivalent_Shrunk_Body_Weight = (478 * self.Shrunk_Body_Weight) / self.Adjusted_Final_Body_Weight
        else: # For Finishing/Growing
            self.Equivalent_Shrunk_Body_Weight = (478 * self.Shrunk_Body_Weight) / self.Adjusted_Final_Body_Weight
        if self.Equivalent_Shrunk_Body_Weight >= 478:
            self.Equivalent_Shrunk_Body_Weight = 478


    def dry_matter_intake(self, NEm_diet, relative_DMI=100):
        """
        Calculates the cow's dry matter intake (DMI) based on CVDS model equations.
        """
        Body_fat_adjustment_factor =  0.7714 + 0.00196 * self.Equivalent_Shrunk_Body_Weight - 0.00000371 * (self.Equivalent_Shrunk_Body_Weight**2)
        DMI_adjustment_factor = 1.15
        Mud_adjustment_factor = 1
        ImplantsFactor = 0.94 if self.usingImplants else 1
        HolsteinFactor = 1.08 if self.holsteinBreeding else 1

        return ((((self.Shrunk_Body_Weight**0.75)*((0.2435 * NEm_diet) - (0.0466 * NEm_diet**2) - 0.1128)) / NEm_diet)
                    * Body_fat_adjustment_factor * DMI_adjustment_factor * Mud_adjustment_factor
                    *ImplantsFactor * HolsteinFactor * (relative_DMI/100))

    def fasting_heat_production_coefficient(self):
        a1 = 0.07 if self.cattle_type == "TRUE" else 0.078
        return a1

    def maintenance_adjustment_acclimatization(self, previous_temperature = 0):
        a2 = ((88.426 - 0.785 * previous_temperature + 0.0116 * previous_temperature**2) - 77)/1000
        return a2

    def net_energy_maintainence( self, Activity_factor = 1.1):
        Activity_factor = random.uniform(1, 1.1)
        Growth_Adjustment_Factor =  0.8 + (self.Body_Condition_Score - 1)*0.05
        Net_Energy_Maintainence_Cold_Stress = 1.07
        Net_Energy_Maintainence_Heat_Stress = 1.07
        a1 = self.fasting_heat_production_coefficient()
        a2 = self.maintenance_adjustment_acclimatization()
        #Net_Energy_Maintainence = (((self.Shrunk_Body_Weight**0.75)*(a1 *Growth_Adjustment_Factor + a2) + Activity_factor + Net_Energy_Maintainence_Cold_Stress)*Net_Energy_Maintainence_Heat_Stress)
        Net_Energy_Maintainence = ((self.Shrunk_Body_Weight**0.75) * (a1 * Growth_Adjustment_Factor * Activity_factor))
        return Net_Energy_Maintainence

class CVDSModel(mesa.Model):
    """
    The CVDSModel simulates a cattle feeding system based on the Cornell Value Discovery System (CVDS).
    It manages cattle agents in pens, tracks their growth, and updates diet plans dynamically.
    """

    def __init__(self, init_cattles, feed_plan, n_pens, seed=42):
        """
        Initializes the CVDS model.

        Args:
            init_cattles (list): List of dictionaries containing initial cattle data.
            feed_plan (list): List of dictionaries specifying diet plans for each pen.
            n_pens (int): Number of pens in the facility.
            seed (int, optional): Seed for randomization to ensure reproducibility.
        """
        super().__init__(seed=seed)
        n = n_pens
        # Simulation day counter
        self.day = 0

        # Data collector to track agent metrics (e.g., weight, dry matter intake)
        self.datacollector = DataCollector(
            agent_reporters={
                "Shrunken_Body_Weight": "Shrunk_Body_Weight",  # Tracks weight changes
                "DMI": "dmi", # Tracks dry matter intake
                "EBF": "Empty_Body_Fat"
            }
        )

        # Number of pens in the feedlot
        self.n = n_pens

        # Number of cattle agents
        self.num_agents = len(init_cattles)

        # Create a multi-grid space (pens) to manage cattle locations
        self.pens = MultiGrid(self.n, 1, torus=False, property_layers=[])

        # Add property layers to store Net Energy for Maintenance (NEm) and Gain (NEg)
        self.pens.add_property_layer(PropertyLayer("NEm_diet", self.n, 1, default_value=0.0, dtype=float))
        self.pens.add_property_layer(PropertyLayer("NEg_diet", self.n, 1, default_value=0.0, dtype=float))

        # Store the initial feed plan
        self.feed_plan = feed_plan

        # Define date format for parsing feeding schedules
        self.date_format = "%m/%d/%Y"

        # Set simulation start date based on the first cattle's intake date
        self.initial_date = datetime.strptime(init_cattles[0]['iDate'],self.date_format)
        self.current_date = datetime.strptime(init_cattles[0]['iDate'],self.date_format)

        # Initialize cattle agents and place them in their respective pens
        for cattle in init_cattles:
            cow = Cow(self, cattle)
            self.pens.place_agent(cow, (int(cattle["PenID"]) - 1, 0))  # Adjusted for 0-based indexing

    def step(self, feed=None):
        """
        Advances the simulation by one day. Updates feed schedules and processes cattle feeding.

        Args:
            feed (optional): Placeholder for future feeding customization.
        """
        # Increment simulation time
        self.current_date = self.current_date + timedelta(days=1)
        self.day += 1
        # print(f'Day: {self.day}')

        # Iterate through all pens to update feed and process cattle growth
        for pen in range(self.n):
            # Identify the correct diet row for the current pen
            for feed_row in range(len(self.feed_plan)):
                if pen + 1 == int(self.feed_plan[feed_row]["PenRecord"]):  # Match pen ID
                    break  # Found the relevant feed row

            # Check if it's time to update the feed plan (assuming sorted dates)
            if datetime.strptime(self.feed_plan[feed_row + 1]['iDate'], self.date_format) == self.current_date:
                #to do: assuming the dates are sorted
                del self.feed_plan[feed_row]  # Remove the past diet entry

            # Update diet properties for the pen
            self.pens.properties["NEm_diet"].set_cell((pen, 0), self.feed_plan[feed_row]["NEm"])
            self.pens.properties["NEg_diet"].set_cell((pen, 0), self.feed_plan[feed_row]["NEg"])

            # Process feeding for all cattle in this pen
            for cow in self.pens.get_cell_list_contents((pen, 0)):
                cow.feed()

        # Collect data at the end of the step for analysis
        self.datacollector.collect(self)


# Define a function to calculate the new value based on DMI
def Emissions(dmi_value):
    """
    This function calculates a new value based on the provided DMI(Kg) value.
    Replace this with your actual calculation logic.
    """
    # Example calculation (replace with your actual logic)
    methane = dmi_value * 14.8 + 16.5
    return methane
