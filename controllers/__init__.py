"""Vent/irrigation controllers for the avinya-twin polyhouse simulator.

Each controller exposes ``vent_policy`` and ``irrigation_policy`` bound
methods with the exact signature ``sim.engine.run`` expects:

    vent_policy(state, soil, weather_row, timestamp) -> float in [0, 1]
    irrigation_policy(state, soil, weather_row, timestamp) -> float, mm

so any controller instance can be plugged straight in:

    run(weather_df, controller.vent_policy, controller.irrigation_policy)
"""
