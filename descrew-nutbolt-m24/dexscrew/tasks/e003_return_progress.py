"""Opt-in actor-command correction, distinct from the physical phase timer.

Late return's saturated time-based command can make the frozen actor drive
away from its reference. Keep its entry command while actual contact/sector
criteria determine completion. Phase timers and all termination gates remain.
This is an explicit local control adaptation, not a reward or encoder reset.
"""


def instrument_return_entry_clock(base):
    class ReturnEntryClock(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            c = self.e002_cycle
            assert c is not None
            original = c.command
            def command():
                result = original()
                result[c.phase == 2, 4] = 0
                return result
            c.command = command
            self.return_progress_receipt = dict(
                actor_return_clock=0, other_phase_clocks='original',
                physical_timer_reset=False, termination_gates_changed=False,
                phase_completion='original measured contact/return sector')
    return ReturnEntryClock
