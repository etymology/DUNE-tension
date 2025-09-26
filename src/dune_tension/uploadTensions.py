from m2m.common import ConnectToAPI, EditAction

# from tensiometer import Tensiometer
import pandas as pd


def load_tension_summary(apa_name: str, layer: str) -> tuple[list, list]:
    datapath = f"data/tension_summaries/tension_summary_{apa_name}_{layer}.csv"
    df = pd.read_csv(datapath, encoding="utf-8")
    if "A" not in df.columns or "B" not in df.columns:
        return "⚠️ File missing required columns 'A' and 'B'", [], []

    # Convert columns to lists, preserving NaNs if present
    a_list = df["A"].tolist()
    b_list = df["B"].tolist()
    return a_list, b_list


def uploadTensions(apa_name: str, layer: str, create_layer_action_id :str) -> None:
    connection, headers = ConnectToAPI()

    # actionTypeFormID = "x_tension_testing"  # This name is misleading: the action type form is the same for ALL LAYERS
    # componentUUID = t.get_uuid()  # This is the identity of APA

    # actionData = {
    #     "apaLayer": t.layer,
    #     "location": "Chicago",
    #     "measurementSystem": "Laser #1",
    #     "date": str(datetime.now()),
    # }

    # create_layer_action_id = PerformAction(
    #     actionTypeFormID, componentUUID, actionData, connection, headers
    # )

    # print(f" Successfully performed action with ID: {create_layer_action_id}")

    tensions_sideA, tensions_sideB = load_tension_summary(apa_name, layer)

    actionData_fields = [
        "measuredTensions_sideA",
        "measuredTensions_sideB",
        "replacedWireSegs",
        "comments",
    ]
    actionData_values = [
        tensions_sideA,
        tensions_sideB,
        "All wire segments are within specified tolerance.",
        "This is an existing single layer tension measurements action, edited via M2M",
    ]

    edit_layer_action_id = EditAction(
        create_layer_action_id,
        actionData_fields,
        actionData_values,
        connection,
        headers,
    )
    print(f" Successfully edited action with ID: {edit_layer_action_id}")

    connection.close()


if __name__ == "__main__":
    uploadTensions("USAPA9*", "G",r"68bf085507c9af803121a611")
