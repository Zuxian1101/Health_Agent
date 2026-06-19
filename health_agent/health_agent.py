class HealthAgent:

    def __init__(self):

        self.state_history = []

    def analyze(self, state):

        hr = state.get("hr")
        emotion = state.get("emotion")
        fatigue = state.get("fatigue")
        work_time = state.get("work_time", 0)

        self.state_history.append(state)

        if len(self.state_history) > 200:
            self.state_history.pop(0)

        if fatigue is not None and fatigue > 75:
            return "Take a 5-minute break"

        if hr is not None and hr > 100:
            return "Relax and breathe"

        if emotion in ["sad", "angry", "fear"]:
            return "You seem stressed. Take a short walk"

        if work_time > 60:
            return "You have been sitting for too long. Stand up"

        return "You are doing fine"

