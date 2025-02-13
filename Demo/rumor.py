# This demo is based on the Repast Simphony tutorial 2: The Rumor Network Model
# https://repast.github.io/repast4py.site/guide/user_guide.html#_tutorial_2_the_rumor_network_model

import ast
from functools import partial
import networkx as nx
import numpy as np
from typing import Dict
from dataclasses import dataclass
from mpi4py import MPI
from repast4py import context as ctx
from repast4py import core, schedule, logging, parameters
from repast4py.network import read_network
from network import *

model = None


class RumorAgent(core.Agent):

    def __init__(self, nid: int, agent_type: int, rank: int, received_rumor=False, shadow=None):
        super().__init__(nid, agent_type, rank)
        self.received_rumor = received_rumor
        self.shadow = shadow or {}

    def save(self):
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its state using this
        method, and any ghost agents of this agent will
        be updated with that data (self.received_rumor).

        Returns:
            The agent's state
        """
        return (self.uid, self.received_rumor, self.shadow)

    def update(self, data: bool, shadow_data: Dict):
        """Updates the state of this agent when it is a ghost
        agent on some rank other than its local one.

        Args:
            data: the new agent state (received_rumor)
        """
        received_rumor = data

        if not self.received_rumor and received_rumor:
            # only update if the received rumor state
            # has changed from false to true
            model.rumor_spreaders.append(self)
            self.received_rumor = received_rumor
        
        self.shadow = shadow_data


def create_rumor_agent(nid, agent_type, rank, **kwargs):
    shadow_data = {}
    if 'data' in kwargs:  # New compressed format
        shadow_data = decompress_and_convert_shadow_data(kwargs['data'])
    return RumorAgent(nid, agent_type, rank, received_rumor=None, shadow=shadow_data)


def restore_agent(agent_data):
    uid = agent_data[0]
    received_rumor = agent_data[1]
    shadow = agent_data[2] if len(agent_data) > 2 else {}  # Handle cases where shadow might not be saved
    return RumorAgent(uid[0], uid[1], uid[2], received_rumor, shadow)


@dataclass
class RumorCounts:
    total_rumor_spreaders: int
    new_rumor_spreaders: int


class Model:
    def __init__(self, comm, params):
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        fpath = params['network_file']
        self.context = ctx.SharedContext(comm)
        read_network(fpath, self.context, create_rumor_agent, restore_agent)
        self.net = self.context.get_projection('rumor_network')

        self.rumor_spreaders = []
        self.rank = comm.Get_rank()
        self._seed_rumor(params['initial_rumor_count'], comm, params["seed"])

        rumored_count = len(self.rumor_spreaders)
        self.counts = RumorCounts(rumored_count, rumored_count)
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params['counts_file'])
        self.data_set.log(0)

        self.rumor_prob = params['rumor_probability']

        # Schedule layer-specific steps
        layer_schedules = params['layer_schedules']
        for layer_id, schedule_config in enumerate(layer_schedules):
            start = schedule_config['start']
            interval = schedule_config['interval']
            self.runner.schedule_repeating_event(start, interval, partial(self.new_step, layer=layer_id))
        
            
    def _seed_rumor(self, init_rumor_count: int, comm, seed = None):
        world_size = comm.Get_size()
        rumor_counts = np.zeros(world_size, dtype=np.int32)
        if self.rank == 0:
            rng = np.random.default_rng()
            if seed is None:
                for _ in range(init_rumor_count):
                    idx = rng.integers(0, high=world_size)
                    rumor_counts[idx] += 1
            else:
                for _ in range(init_rumor_count - len(seed)):
                    idx = rng.integers(0, high=world_size)
                    rumor_counts[idx] += 1

        rumor_count = np.empty(1, dtype=np.int32)
        comm.Scatter(rumor_counts, rumor_count, root=0)

        for agent in self.context.agents(count=rumor_count[0], shuffle=True):
            agent.received_rumor = True
            self.rumor_spreaders.append(agent)

        if seed is not None:
            for agent in self.context.agents():
                if agent.uid[0] in seed:
                    agent.received_rumor = True
                    self.rumor_spreaders.append(agent)
                    
    def at_end(self):
        self.data_set.close()

    def new_step(self, layer):
        #print(f"Rank {self.rank} is executing step {self.runner.schedule.tick} for layer {layer}")
        new_rumor_spreaders = []
        rng = np.random.default_rng()
        for agent in self.rumor_spreaders:
            # Agent does not have outgoing edges in this layer
            if layer not in agent.shadow.keys():
                continue
            ngh_tuples = agent.shadow[layer].keys()
            for ngh_tuple in ngh_tuples:
                ngh_agent = self.context.agent(ngh_tuple)
                if ngh_agent is None:
                    continue  # Neighbor not found (shouldn't happen if network is correct)
                # Only spread to local agents that haven't received the rumor
                if ngh_agent.local_rank == self.rank and not ngh_agent.received_rumor:
                    if rng.uniform() <= self.rumor_prob:
                        ngh_agent.received_rumor = True
                        new_rumor_spreaders.append(ngh_agent)
        # Update the list of rumor spreaders with new local agents
        self.rumor_spreaders += new_rumor_spreaders
        # Update counts
        self.counts.new_rumor_spreaders = len(new_rumor_spreaders)
        self.counts.total_rumor_spreaders += self.counts.new_rumor_spreaders
        self.data_set.log(self.runner.schedule.tick)
        # Synchronize agents across ranks
        self.context.synchronize(restore_agent)

    def start(self):
        self.runner.execute()


def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)

