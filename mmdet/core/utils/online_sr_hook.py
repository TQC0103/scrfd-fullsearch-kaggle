import json
import os

import numpy as np
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class OnlineSRSchedulerHook(Hook):
    """Adjust RandomSquareCrop probabilities from stride-wise train stats."""

    def __init__(self,
                 state_file='sr_scheduler_state.json',
                 target_strides=(8, 16, 32),
                 target_positive_ratios=(0.5, 0.3, 0.2),
                 loss_weight=0.65,
                 deficit_weight=0.35,
                 update_momentum=0.6,
                 temperature=0.8,
                 min_crop_prob=0.03,
                 log_interval=1):
        self.state_file = state_file
        self.target_strides = tuple(int(s) for s in target_strides)
        self.target_positive_ratios = self._normalize(target_positive_ratios)
        self.loss_weight = float(loss_weight)
        self.deficit_weight = float(deficit_weight)
        self.update_momentum = float(update_momentum)
        self.temperature = max(float(temperature), 1e-4)
        self.min_crop_prob = float(min_crop_prob)
        self.log_interval = max(1, int(log_interval))
        self._crop_choice = None
        self._crop_probs = None

    def before_run(self, runner):
        crop_choice = self._find_crop_choice(runner)
        if crop_choice is None:
            runner.logger.warning(
                'OnlineSRSchedulerHook could not find RandomSquareCrop in the '
                'train pipeline. The hook will stay inactive.')
            return

        self._crop_choice = [float(x) for x in crop_choice]
        self._crop_probs = self._uniform(len(self._crop_choice))

        state_path = self._state_path(runner)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        self._write_state(
            state_path,
            {
                'epoch': 0,
                'crop_choice': self._crop_choice,
                'crop_choice_weights': self._crop_probs,
                'stride_metrics': {},
            })

    def before_train_epoch(self, runner):
        head = self._get_bbox_head(runner)
        if head is not None and hasattr(head, 'reset_sr_epoch_stats'):
            head.reset_sr_epoch_stats()

    def after_train_epoch(self, runner):
        if self._crop_choice is None:
            return

        head = self._get_bbox_head(runner)
        if head is None or not hasattr(head, 'get_sr_epoch_stats'):
            return

        stats = head.get_sr_epoch_stats()
        if not stats:
            return

        pos_ratios, loss_ratios = self._extract_ratios(stats)
        urgency = []
        for idx in range(len(self.target_strides)):
            deficit = max(0.0, self.target_positive_ratios[idx] - pos_ratios[idx])
            urgency.append(self.loss_weight * loss_ratios[idx] +
                           self.deficit_weight * deficit)

        raw_scores = []
        for scale in self._crop_choice:
            prefs = self._scale_preferences(scale)
            raw_scores.append(float(np.dot(prefs, urgency)))

        fresh_probs = self._softmax(raw_scores)
        mixed_probs = [
            self.update_momentum * old + (1.0 - self.update_momentum) * new
            for old, new in zip(self._crop_probs, fresh_probs)
        ]
        self._crop_probs = self._with_floor(mixed_probs)

        state_path = self._state_path(runner)
        self._write_state(
            state_path,
            {
                'epoch': int(runner.epoch) + 1,
                'crop_choice': self._crop_choice,
                'crop_choice_weights': self._crop_probs,
                'stride_metrics': {
                    str(stride): {
                        'pos_ratio': float(pos_ratios[idx]),
                        'loss_ratio': float(loss_ratios[idx]),
                    }
                    for idx, stride in enumerate(self.target_strides)
                }
            })

        if (runner.epoch + 1) % self.log_interval == 0:
            summary = ', '.join(
                f'{choice:.2f}:{prob:.3f}'
                for choice, prob in zip(self._crop_choice, self._crop_probs))
            runner.logger.info(
                'Online SR scheduler updated crop probabilities after epoch %d: %s',
                runner.epoch + 1,
                summary)

    def _state_path(self, runner):
        del runner
        return os.path.abspath(self.state_file)

    def _find_crop_choice(self, runner):
        dataset = getattr(runner.data_loader, 'dataset', None)
        pipeline = getattr(dataset, 'pipeline', None)
        transforms = getattr(pipeline, 'transforms', [])
        for transform in transforms:
            if type(transform).__name__ == 'RandomSquareCrop':
                return getattr(transform, 'crop_choice', None)
        return None

    def _get_bbox_head(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        return getattr(model, 'bbox_head', None)

    def _extract_ratios(self, stats):
        pos_values = [float(stats['pos_counts'].get(stride, 0.0))
                      for stride in self.target_strides]
        loss_values = [float(stats['loss_sums'].get(stride, 0.0))
                       for stride in self.target_strides]

        pos_total = sum(pos_values)
        loss_total = sum(loss_values)
        pos_ratios = self._uniform(len(pos_values)) if pos_total <= 0 else \
            [value / pos_total for value in pos_values]
        loss_ratios = self._uniform(len(loss_values)) if loss_total <= 0 else \
            [value / loss_total for value in loss_values]
        return pos_ratios, loss_ratios

    def _scale_preferences(self, scale):
        delta = float(scale) - 1.0
        prefer_stride8 = 0.2 + max(0.0, delta) * 1.5
        prefer_stride32 = 0.2 + max(0.0, -delta) * 1.5
        prefer_stride16 = 0.2 + max(0.0, 1.0 - abs(delta) / 0.6)
        return np.asarray(
            self._normalize([prefer_stride8, prefer_stride16, prefer_stride32]),
            dtype=np.float32)

    def _softmax(self, values):
        values = np.asarray(values, dtype=np.float32) / self.temperature
        values -= values.max()
        probs = np.exp(values)
        probs_sum = probs.sum()
        if not np.isfinite(probs_sum) or probs_sum <= 0:
            return self._uniform(len(values))
        probs /= probs_sum
        return probs.tolist()

    def _with_floor(self, probs):
        min_prob = max(0.0, min(self.min_crop_prob, 1.0 / max(len(probs), 1)))
        probs = np.asarray(self._normalize(probs), dtype=np.float32)
        probs = np.maximum(probs, min_prob)
        probs /= probs.sum()
        return probs.tolist()

    def _uniform(self, length):
        if length <= 0:
            return []
        return [1.0 / length for _ in range(length)]

    def _normalize(self, values):
        values = [max(float(v), 1e-8) for v in values]
        total = sum(values)
        if total <= 0:
            return self._uniform(len(values))
        return [value / total for value in values]

    def _write_state(self, state_path, state):
        tmp_path = f'{state_path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, state_path)
