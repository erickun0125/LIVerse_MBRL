import torch
import torch.nn.functional as F


class MPCPlanner:
    """Model-predictive control planner with cross-entropy method and learned transition model."""

    def __init__(self, transition_model, reward_function, planning_horizon, optimization_iters,
                 candidates, top_candidates, action_size, min_action, max_action, device, reward_form="similarity"):
        self.transition_model = transition_model
        self.reward_function = reward_function
        self.planning_horizon = planning_horizon
        self.optimization_iters = optimization_iters
        self.candidates = candidates
        self.top_candidates = top_candidates
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.device = device
        self.reward_form = reward_form

    def plan(self, belief, state, target_embedding):
        """Plan action sequence using CEM and return first action."""
        B, H, Z = belief.size(0), belief.size(1), state["stoch"].size(1)

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=self.device)
        action_std_dev = torch.ones(self.planning_horizon, B, 1, self.action_size, device=self.device)

        # Expand belief and state for batch processing
        belief_expanded = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H)
        state_expanded = {k: v.unsqueeze(dim=1).expand(B, self.candidates, v.size(1), *v.shape[2:]).reshape(-1, v.size(1), *v.shape[2:])
                          for k, v in state.items()}

        # CEM optimization loop
        for _ in range(self.optimization_iters):
            actions = (action_mean + action_std_dev * torch.randn(
                self.planning_horizon, B, self.candidates, self.action_size, device=self.device
            )).view(self.planning_horizon, B * self.candidates, self.action_size)

            actions = torch.clamp(actions, min=self.min_action, max=self.max_action)

            returns = torch.zeros(B * self.candidates, device=self.device)

            curr_belief = belief_expanded.clone()
            curr_state = {k: v.clone() for k, v in state_expanded.items()}

            feat = self.transition_model.get_feat(curr_state)
            prev_similarities = self.reward_function(feat, curr_state, actions[0], target_embedding)

            for t in range(self.planning_horizon):
                action = actions[t]

                next_beliefs, _, _, _, next_states, _, _ = self.transition_model(
                    curr_state, action.unsqueeze(0), curr_belief.unsqueeze(0), None, None)

                curr_belief = next_beliefs.squeeze(0)
                curr_state = {k: v.squeeze(0) for k, v in next_states.items()}

                feat = self.transition_model.get_feat(curr_state)
                current_similarities = self.reward_function(feat, curr_state, action, target_embedding)

                if self.reward_form == "difference":
                    reward = current_similarities - prev_similarities
                else:
                    reward = current_similarities

                prev_similarities = current_similarities.clone()

                returns += reward

            returns = returns.view(B, self.candidates)

            _, topk = returns.topk(self.top_candidates, dim=1, largest=True, sorted=False)
            topk += self.candidates * torch.arange(0, B, device=self.device).unsqueeze(dim=1)

            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)

            action_mean = best_actions.mean(dim=2, keepdim=True)
            action_std_dev = best_actions.std(dim=2, unbiased=False, keepdim=True)

        return action_mean[0].squeeze(dim=1)


def update_belief_and_act(world_model, planner, belief, posterior_state, action, observation, is_first, target_embedding):
    """Update belief and state with new observation, then plan action."""
    obs_processed = world_model.preprocess(observation)

    embed = world_model.encoder(obs_processed).unsqueeze(dim=0)

    belief, _, _, _, posterior_state, _, _ = world_model.dynamics(
        posterior_state, action.unsqueeze(dim=0), belief, embed, is_first)

    belief, posterior_state = belief.squeeze(dim=0), {k: v.squeeze(dim=0) for k, v in posterior_state.items()}

    action = planner.plan(belief, posterior_state, target_embedding)

    return belief, posterior_state, action
