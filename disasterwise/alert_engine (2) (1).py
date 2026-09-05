def compute_alert(damage_percentage):

    
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


damage = float(input("Enter damage percentage: "))

level, msg, dsi = compute_alert(damage)

print("Alert Level:", level)
print("Message:", msg)
print("Disaster Severity Index:", dsi)