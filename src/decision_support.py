"""
decision_support.py
Intelligent Air Traffic Control Decision Support System (DSS).
Generates operational directives, flow management programs, and runway advisories
based on predicted traffic volume, congestion level, and environmental constraints.
"""

from typing import Dict, List, Any

CONGESTION_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
CONGESTION_COLORS = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"} # Green, Yellow, Red

def generate_atc_advisories(
    airport: str,
    current_flights: int,
    predicted_flights: int,
    congestion_level: int,
    confidence_pct: float,
    runway_utilization: float,
    avg_delay: float,
    weather_delay: float,
    capacity: int
) -> Dict[str, Any]:
    """
    Evaluates real-time & predicted metrics to issue tactical and strategic ATC advisories.
    """
    level_name = CONGESTION_LABELS.get(congestion_level, "LOW")
    color = CONGESTION_COLORS.get(congestion_level, "#22c55e")
    
    advisories: List[Dict[str, str]] = []
    flow_actions: List[str] = []
    runway_strategy: str = ""
    alert_level: str = "NORMAL"
    
    # 1. HIGH CONGESTION DIRECTIVES
    if congestion_level == 2 or runway_utilization >= 85.0:
        alert_level = "CRITICAL"
        
        # Ground Delay Program
        estimated_ground_hold = max(15, int(avg_delay * 0.75))
        flow_actions.append(f"Ground Delay Program (GDP): Issue {estimated_ground_hold}-min departure release hold for short-haul flights.")
        
        # Airborne Metering & Holding
        flow_actions.append("Airspace Flow Management: Enact en-route speed reduction (Mach 0.78 -> 0.72) at 120 NM boundary to meter inbound stream.")
        flow_actions.append("Holding Stacks: Prepare holding pattern stacks at outer sector fixes (FL140 - FL200).")
        
        # Runway Strategy
        runway_strategy = "Dual Arrival Preference: Dedicate primary instrument runway to continuous arrivals; batch departures in 4-slot windows."
        
        advisories.append({
            "category": "Flow Management",
            "priority": "HIGH",
            "action": f"Traffic pressure is {runway_utilization:.1f}% capacity. Delay escalation likely in next 30-60 min."
        })
        advisories.append({
            "category": "Apron & Ground Control",
            "priority": "HIGH",
            "action": "Throttle gate pushback clearances by 10 min to avoid taxiway congestion and excessive fuel burn."
        })
        
    # 2. MEDIUM CONGESTION DIRECTIVES
    elif congestion_level == 1 or runway_utilization >= 65.0:
        alert_level = "ELEVATED"
        flow_actions.append("Departure Sequencing: Interleave heavy and medium wake-vortex aircraft to minimize separation buffer.")
        flow_actions.append("Taxi Flow: Direct narrow-body arrivals to rapid-exit taxiways to vacate active runway within 45 seconds.")
        runway_strategy = "Balanced Mixed-Mode: Runways operate simultaneous arrivals and departures with minimum radar separation (3 NM)."
        
        advisories.append({
            "category": "Slot Sequencing",
            "priority": "MEDIUM",
            "action": f"Runway utilization at {runway_utilization:.1f}%. Traffic volume near peak throughput threshold."
        })
        advisories.append({
            "category": "Ground Movement",
            "priority": "MEDIUM",
            "action": "Activate secondary parallel taxiway to prevent aircraft standoff at Terminal gates."
        })
        
    # 3. LOW CONGESTION DIRECTIVES
    else:
        alert_level = "NOMINAL"
        flow_actions.append("Green Operations: Continuous Descent Operations (CDO/CDA) authorized for optimal fuel conservation.")
        flow_actions.append("Direct Routing: Clearance Delivery can issue direct-to fixes on standard instrument departures (SIDs).")
        runway_strategy = "Standard Operations: Single runway or independent operations without flow restrictions."
        
        advisories.append({
            "category": "Efficiency Advisory",
            "priority": "LOW",
            "action": f"Traffic flow is smooth at {runway_utilization:.1f}% capacity. No holding or slot restrictions required."
        })

    # Weather impact advisory
    if weather_delay > 15.0:
        flow_actions.insert(0, f"Weather Advisory: Adverse weather causing ~{weather_delay:.0f} min systemic delay. Increase runway separation to 5 NM.")
        advisories.insert(0, {
            "category": "Meteorological Alert",
            "priority": "CRITICAL",
            "action": "Low visibility / convective weather procedures active. Category II/III ILS operations enforced."
        })

    return {
        "airport": airport,
        "alert_level": alert_level,
        "congestion_level": level_name,
        "color": color,
        "confidence": confidence_pct,
        "current_flights": current_flights,
        "predicted_flights": predicted_flights,
        "capacity": capacity,
        "runway_utilization": runway_utilization,
        "flow_actions": flow_actions,
        "runway_strategy": runway_strategy,
        "advisories": advisories
    }
