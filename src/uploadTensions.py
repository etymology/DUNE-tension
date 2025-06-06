from m2m.common import ConnectToAPI, EditAction
from dune_tension.tensiometer import Tensiometer


def uploadTensions(t: Tensiometer):
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

    create_layer_action_id = r"683f72822967cf595ddbd6d3"
    print(f" Successfully performed action with ID: {create_layer_action_id}")

    tensions_sideA, tensions_sideB = t.load_tension_summary()

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
    t = Tensiometer(
        apa_name="US_APA9",
        layer="X",
        side="A",
    )

    uploadTensions(t)
