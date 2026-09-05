def compute_alert(damage_percentage):
    """
    Computes alert level based on damage percentage.
    DSI is Disaster Severity Index.
    """
    DSI = damage_percentage * 1.5

    if damage_percentage < 20:
        alert_level = "GREEN"
        message = "Minor damage detected."
    elif damage_percentage < 50:
        alert_level = "YELLOW"
        message = "Moderate damage detected."
    else:
        alert_level = "RED"
        message = "Severe damage detected."

    return alert_level, message, DSI
