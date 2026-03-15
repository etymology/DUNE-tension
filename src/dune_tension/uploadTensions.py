try:  # pragma: no cover - fallback for local script execution
    from dune_tension.m2m.common import ConnectToAPI, EditAction
    from dune_tension.summaries import get_expected_range, get_tension_series
    from dune_tension.tensiometer_functions import make_config
except ImportError:  # pragma: no cover
    from m2m.common import ConnectToAPI, EditAction
    from summaries import get_expected_range, get_tension_series  # type: ignore
    from tensiometer_functions import make_config  # type: ignore


def load_tension_summary(apa_name: str, layer: str) -> tuple[list, list]:
    config = make_config(apa_name=apa_name, layer=layer, side="A")
    tension_series = get_tension_series(config)
    wire_range = list(get_expected_range(layer))
    nan = float("nan")
    return (
        [tension_series["A"].get(wire, nan) for wire in wire_range],
        [tension_series["B"].get(wire, nan) for wire in wire_range],
    )


def uploadTensions(apa_name: str, layer: str, create_layer_action_id: str) -> None:
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
    print(f" Uploading {len(tensions_sideA)} tensions for APA {apa_name} layer {layer}...")
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
    uploadTensions("USAPA11", "G", r"698392969c0b3b26e5f29f47")
