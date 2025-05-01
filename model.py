
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import csv
import mesa
import random

from mesa.space import MultiGrid, PropertyLayer
from datetime import datetime, timedelta
from mesa.datacollection import DataCollector


class Cow(mesa.Agent):

    def __init__(self, model, cattle_info):
        super().__init__(model)

        self.hip_height = float(cattle_info["HipHeight"])
        self.age = float(cattle_info["AgeHipHeight"]) * 30
        self.cattle_type = cattle_info["Beef"]
        self.gender = cattle_info["Sex"]
        self.full_body_weight = float(cattle_info["iBW"])
        self.Shrunk_Body_Weight = (
            self.full_body_weight if cattle_info["IsiBWShrunk"] == "TRUE" else 0.94 * self.full_body_weight
        )

        self.Adjusted_Final_Body_Weight = 0.0
        self.usingImplants = cattle_info["Implants"] == "TRUE"
        self.holsteinBreeding = cattle_info["Holstein"] == "TRUE"
        self.dmi = 0.0
        self.Body_Condition_Score = int(cattle_info["BCS"])
        self.Feed_Required = 0.0
        self.Retained_Energy = 0.0

        if self.gender == "B":
            self.frame_score = -11.5480 + 0.4878 * self.hip_height - 0.0289 * self.age + 0.00001947 * self.age**2 + 0.0000334 * self.hip_height * self.age
            self.Adjusted_Final_Body_Weight = 33.35 * self.frame_score + 366.52
        else:
            self.frame_score = -11.7086 + 0.4723 * self.hip_height - 0.0239 * self.age + 0.00001460 * self.age**2 + 0.0000759 * self.hip_height * self.age
            self.Adjusted_Final_Body_Weight = 26.70 * self.frame_score + 293.2

        self.Equivalent_Shrunk_Body_Weight = (478 * self.Shrunk_Body_Weight) / self.Adjusted_Final_Body_Weight
        if self.Equivalent_Shrunk_Body_Weight >= 478:
            self.Equivalent_Shrunk_Body_Weight = 478

        self.Fat = 0.0
        self.YG = 5
        self.Shrunken_Weight_Gain = 0.0
        self.Empty_Weight_Gain = 0.0
        self.Fat_In_EWG = 0.0
        self.Expected_EBF = 4.749 + 7.861 * self.YG - 8.006 * (1.052 - 0.0317 * self.YG + 0.0051 * self.YG)**0.5
        self.Empty_Body_Weight = 0.891 * self.Shrunk_Body_Weight
        self.Empty_Body_Fat = 0.244 * self.Empty_Body_Weight - 15.4135
        self.Equivalent_Carcass_Weight = (0.891 * self.Equivalent_Shrunk_Body_Weight - 30.26) / 1.36
        self.Carcass_Weight_Percent = self.Equivalent_Carcass_Weight / self.Equivalent_Shrunk_Body_Weight
        self.Carcass_Weight = self.Carcass_Weight_Percent * self.Shrunk_Body_Weight
        self.Carcass_Weight_Gain = 0.0

    def feed(self):
        pen_id, _ = self.pos
        NEm_diet = self.model.pens.properties["NEm_diet"].data[pen_id, 0]
        NEg_diet = self.model.pens.properties["NEg_diet"].data[pen_id, 0]

        self.dmi = self.dry_matter_intake(NEm_diet) * 2 + 5
        self.NEm_req = self.net_energy_maintainence()
        self.Feed_Required = self.NEm_req / (NEm_diet * 1.12)

        self.Retained_Energy = (self.dmi - self.Feed_Required) * NEg_diet
        self.Retained_Energy = max(0, self.Retained_Energy)

        self.Shrunken_Weight_Gain = 13.91 * (self.Equivalent_Shrunk_Body_Weight ** -0.6837) * self.Retained_Energy**0.9116
        self.Shrunk_Body_Weight += self.Shrunken_Weight_Gain

        self.Empty_Weight_Gain = 0.956 * self.Shrunken_Weight_Gain
        self.Fat_In_EWG = 0.122 * self.Retained_Energy / self.Empty_Weight_Gain - 0.146
        self.Fat += self.Fat_In_EWG * self.Empty_Weight_Gain
        self.Empty_Body_Weight = 0.891 * self.Shrunk_Body_Weight
        self.Empty_Body_Fat = self.Fat * 100 / self.Empty_Body_Weight

        self.Equivalent_Carcass_Weight = (0.891 * self.Equivalent_Shrunk_Body_Weight - 30.26) / 1.36
        self.Carcass_Weight_Percent = self.Equivalent_Carcass_Weight / self.Equivalent_Shrunk_Body_Weight
        new_Carcass_Weight = self.Carcass_Weight_Percent * self.Shrunk_Body_Weight

        self.Carcass_Weight_Gain = new_Carcass_Weight - self.Carcass_Weight
        self.Carcass_Weight = new_Carcass_Weight

        if self.gender == "B":
            self.frame_score = -11.5480 + 0.4878 * self.hip_height - 0.0289 * self.age + 0.00001947 * self.age**2 + 0.0000334 * self.hip_height * self.age
            self.Adjusted_Final_Body_Weight = 33.35 * self.frame_score + 366.52
        else:
            self.frame_score = -11.7086 + 0.4723 * self.hip_height - 0.0239 * self.age + 0.00001460 * self.age**2 + 0.0000759 * self.hip_height * self.age
            self.Adjusted_Final_Body_Weight = 26.70 * self.frame_score + 293.2

        self.Equivalent_Shrunk_Body_Weight = (478 * self.Shrunk_Body_Weight) / self.Adjusted_Final_Body_Weight
        if self.Equivalent_Shrunk_Body_Weight >= 478:
            self.Equivalent_Shrunk_Body_Weight = 478

        self.emission = self.Emissions(self.dmi)

    def dry_matter_intake(self, NEm_diet, relative_DMI=100):
        Body_fat_adjustment_factor = 0.7714 + 0.00196 * self.Equivalent_Shrunk_Body_Weight - 0.00000371 * (self.Equivalent_Shrunk_Body_Weight ** 2)
        DMI_adjustment_factor = 1.15
        Mud_adjustment_factor = 1
        ImplantsFactor = 0.94 if self.usingImplants else 1
        HolsteinFactor = 1.08 if self.holsteinBreeding else 1

        return ((((self.Shrunk_Body_Weight**0.75) * ((0.2435 * NEm_diet) - (0.0466 * NEm_diet**2) - 0.1128)) / NEm_diet)
                * Body_fat_adjustment_factor * DMI_adjustment_factor * Mud_adjustment_factor
                * ImplantsFactor * HolsteinFactor * (relative_DMI / 100))

    def fasting_heat_production_coefficient(self):
        a1 = 0.07 if self.cattle_type == "TRUE" else 0.078
        return a1

    def maintenance_adjustment_acclimatization(self, previous_temperature=0):
        a2 = ((88.426 - 0.785 * previous_temperature + 0.0116 * previous_temperature**2) - 77) / 1000
        return a2

    def net_energy_maintainence(self, Activity_factor=1.1):
        Activity_factor = random.uniform(1, 1.1)
        Growth_Adjustment_Factor = 0.8 + (self.Body_Condition_Score - 1) * 0.05
        a1 = self.fasting_heat_production_coefficient()
        return ((self.Shrunk_Body_Weight**0.75) * (a1 * Growth_Adjustment_Factor * Activity_factor))

    def Emissions(self, dmi_value):
        methane = dmi_value * 14.8 + 16.5
        return methane

class CVDSModel(mesa.Model):

    def __init__(self, init_cattles, feed_plan, n_pens, seed=42):
        super().__init__(seed=seed)
        self.day = 0
        self.datacollector = DataCollector(
            agent_reporters={
                "Shrunken_Body_Weight": "Shrunk_Body_Weight",
                "DMI": "dmi",
                "EBF": "Empty_Body_Fat",
                "Emissions(g)": "emission"
            }
        )
        self.n = n_pens
        self.num_agents = len(init_cattles)
        self.pens = MultiGrid(self.n, 1, torus=False, property_layers=[])
        self.pens.add_property_layer(PropertyLayer("NEm_diet", self.n, 1, default_value=0.0, dtype=float))
        self.pens.add_property_layer(PropertyLayer("NEg_diet", self.n, 1, default_value=0.0, dtype=float))
        self.feed_plan = feed_plan
        self.date_format = "%m/%d/%Y"
        self.initial_date = datetime.strptime(init_cattles[0]['iDate'], self.date_format)
        self.current_date = self.initial_date

        for cattle in init_cattles:
            cow = Cow(self, cattle)
            self.pens.place_agent(cow, (int(cattle["PenID"]) - 1, 0))

    def step(self, feed=None):
        self.current_date = self.current_date + timedelta(days=1)
        self.day += 1

        for pen in range(self.n):
            for feed_row in range(len(self.feed_plan)):
                if pen + 1 == int(self.feed_plan[feed_row]["PenRecord"]):
                    break

            if datetime.strptime(self.feed_plan[feed_row + 1]['iDate'], self.date_format) == self.current_date:
                del self.feed_plan[feed_row]

            self.pens.properties["NEm_diet"].set_cell((pen, 0), self.feed_plan[feed_row]["NEm"])
            self.pens.properties["NEg_diet"].set_cell((pen, 0), self.feed_plan[feed_row]["NEg"])

            for cow in self.pens.get_cell_list_contents((pen, 0)):
                cow.feed()

        self.datacollector.collect(self)
