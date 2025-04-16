
from model import *


# Open the CSV file in read mode
with open("./Cattle_Info.csv", 'r') as file:
    csv_reader = csv.DictReader(file)  # Read file as dictionary format
    header = csv_reader.fieldnames  # Extract the header names



# Initialize an empty list to store cattle information
cattles = []

# Open the CSV file in read mode
with open("./Cattle_Info.csv", 'r') as file:
    csv_reader = csv.DictReader(file)  # Read CSV as a dictionary

    header = csv_reader.fieldnames  # Extract column names (header)

    # Iterate through each row in the CSV file and store it in the list
    for row in csv_reader:
        cattles.append(row)  # Append each row (as a dictionary) to the list

# Initialize an empty list to store feed plan data
feed_plan = []

# Open the CSV file in read mode
with open("./Diet_Plan.csv", 'r') as file:
    csv_reader = csv.DictReader(file)  # Read CSV as a dictionary

    header = csv_reader.fieldnames  # Extract column names (header)

    # Iterate through each row in the CSV file and store it in the list
    for row in csv_reader:
        feed_plan.append(row)  # Append each row (as a dictionary) to the list



starter_model = CVDSModel(cattles, feed_plan, 2)

# model = CVDSModel(cattles, feed_plan, 2)
for step in range(140):
    starter_model.step()

df = starter_model.datacollector.get_agent_vars_dataframe()
data = df.reset_index()
data['Emissions'] = data['DMI'].apply(Emissions)

