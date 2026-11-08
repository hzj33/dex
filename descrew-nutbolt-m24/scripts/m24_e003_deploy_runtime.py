"""Shared deploy-student simulation adapter; consumes no phase/contact/angle."""
import torch


def install_deployment_limits(env, policy):
    assert env.target_smoothing_alpha == 0 and env.num_actions == 16
    assert not env.config['env']['controller'].get('target_error_limit', 0)
    env.action_scale = float(policy.action_scale)
    env.xhand_hand_dof_lower_limits = policy.lower.to(env.device)
    env.xhand_hand_dof_upper_limits = policy.upper.to(env.device)


class DeployInference:
    def __init__(self, env, artifact, record_predictions=False, inference_device=None):
        self.inference_device = str(inference_device or env.rl_device)
        self.model = torch.jit.load(str(artifact), map_location=self.inference_device).eval()
        install_deployment_limits(env, self.model)
        self.history = None
        self.record_predictions = record_predictions
        self.prediction = None
        self.env = env
        self.contract_errors = None

    def _eval_select_action(self, obs):
        frame = obs['proprio_hist'][:, -1].to(self.inference_device)
        q, previous = frame[:, :16], frame[:, 16:]
        if self.history is None:
            # Original M24 fills pre-start history with the initialization target;
            # the current measurement is appended only in the control step.
            self.history = self.model.reset_history(previous, previous)
        prior_obs = self.history[:, -3:].reshape(-1, 96)
        target, self.history, action = self.model(q, previous, self.history)
        if self.record_predictions:
            _, logits, clock, _ = self.model.predict(prior_obs, self.history)
            self.prediction = torch.cat((logits.softmax(-1), clock), -1).detach()
            sent = self.env.prev_targets[:, :16].to(self.inference_device)
            simulated_target = torch.maximum(torch.minimum(
                sent + float(self.model.action_scale) * action,
                self.model.upper), self.model.lower)
            self.contract_errors = torch.stack((
                (previous - sent).abs().amax(-1),
                (target - simulated_target).abs().amax(-1)), -1).detach()
        # The simulator applies the same position increment, mask and SDK limits.
        return action, {}
