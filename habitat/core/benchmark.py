#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import os
from collections import defaultdict
from typing import Dict, Optional
import numpy as np

from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from tqdm import tqdm
import json
import os


class Benchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote: bool = False, overriden_args: list = None,
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths, overriden_args)
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-lab repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        time.sleep(60)

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        for _ in tqdm(range(num_episodes)):
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        if agent.args.do_error_analysis:
            with open(agent.args.do_error_analysis, "a") as wf:
                wf.write("\n====\n")
        pbar = tqdm(range(num_episodes), desc="")
        for _ in pbar:
            observations = self._env.reset()
            agent.reset()

            while not self._env.episode_over:
                if agent.args.do_error_analysis:
                    # Add these fields for error analysis
                    observations['origin'] = np.array(self._env.current_episode.start_position)
                    observations['rotation_world_start'] = np.array(self._env.current_episode.start_rotation)
                    observations['gt_goal_positions'] = [np.array(g.position) for g in self._env.current_episode.goals]
                    observations['success_distance'] = self._env.task.measurements.measures['success']._config.SUCCESS_DISTANCE
                    observations['self_position'] = self._env.task._sim.get_agent_state().position
                    observations['distance_to_goal'] = self._env.task.measurements.measures['distance_to_goal'].get_metric()
                action = agent.act(observations)
                observations = self._env.step(action)
                # if self._env.task.measurements.measures['distance_to_goal']._metric:
                # false negative (but what if taret object not yet in sight????)
                # self._env.task._sim.get_agent_state().position

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            if agent.args.do_error_analysis:
                if metrics['success']:
                    # write to file
                    with open(agent.args.do_error_analysis, "a") as wf:
                        wf.write(json.dumps({'envid': self._env.current_episode.episode_id, 'success': True, 'distance': metrics['distance_to_goal'], 'saw_target_frames': action['saw_target'], 'nearby_objs': action['other_objs'], 'target': action['objectgoal']})+"\n")
                else:
                    # failure mode...
                    with open(agent.args.do_error_analysis, "a") as wf:
                        wf.write(json.dumps({'envid': self._env.current_episode.episode_id, 'success': False, 'distance': metrics['distance_to_goal'], 'failures': action['failure_modes'], 'saw_target_frames': action['saw_target'], 'nearby_objs': action['other_objs'], 'target': action['objectgoal']})+"\n")
            count_episodes += 1
            pbar.set_description(' '.join([
                f'{m}={agg_metrics[m] / count_episodes:.2f}'
                if not isinstance(agg_metrics[m], dict)
                else f'{m}={json.dumps({sub_m: agg_metrics[m][sub_m] / count_episodes for sub_m in agg_metrics[m]})}'
                for m in agg_metrics
            ]))

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        if agent.args.do_error_analysis:
            # log metrics
            with open(agent.args.do_error_analysis, "a") as wf:
                wf.write("Metrics: "+json.dumps(avg_metrics)+"\n")

        return avg_metrics

    def evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)
