import numpy as np
import numpy.random as random

class Block_Propagation_Simulator:

  def __init__(self, N, policy, num_blocks, arrival_rate):
    # global parameters
    self.N = N
    self.policy = policy
    self.num_blocks = num_blocks
    self.arrival_rate = arrival_rate
    self.arrival_times = self.__get_arrival_times(self.num_blocks, self.arrival_rate)
    self.arrival_index = 0

    # temporal parameters
    self.time = 0.
    self.dt = 0.
    self.beta = 1.0/self.N

    self.cycle_begin = 0.
    self.cycle_lengths = []
    self.blocks_this_cycle = 0
    self.blocks_per_cycle = []

    self.cycles = 0
    self.num_active_blocks_integral = 0.
    self.num_active_blocks_list = []
    self.num_missing_block_copies_integral = 0.
    self.age_of_information = 0

    # state-based parameters
    self.block_array = np.zeros((0, self.N))
    self.block_sources = np.zeros((0, self.N))
    self.block_times = []

    self.consistent = True
 
  # compute the arrival times for new blocks 
  def __get_arrival_times(self, num_blocks, arrival_rate):
    arrival_beta = 1.0/arrival_rate
    time = 0.
    arrival_times = []
    for n in range(num_blocks):
      time += random.exponential(arrival_beta)
      arrival_times.append(time)
    return arrival_times


  # run the simulator
  def run(self):
    random.seed()

    #while self.cycles < self.num_cycles:
      #self.__event()
    while self.arrival_index < self.num_blocks:
      self.__event()

    # due to a quirk in the way these are calculated during runtime,
    #    we need to add the last data now
    #self.cycle_lengths.append(self.time - self.cycle_begin)
    #self.blocks_per_cycle.append(self.blocks_this_cycle)

    return [self.time, self.num_active_blocks_integral, 
            self.num_missing_block_copies_integral,
            np.mean(self.cycle_lengths),
            np.mean(self.blocks_per_cycle),
            self.age_of_information,
            self.num_active_blocks_list]


  # compute a single arrival or transmission event
  def __event(self):
    #print(self.block_array)

    self.__remove_inactive_blocks()
    #print(self.block_array)
    #input()

    arrival = False

    # update the system time
    self.dt = random.exponential(self.beta)
    if self.consistent:
      arrival = True

      self.cycle_lengths.append(self.time - self.cycle_begin)
      np.append(self.cycle_lengths, self.time - self.cycle_begin)
      self.cycle_begin = self.time

      self.dt = self.arrival_times[self.arrival_index] - self.time
      self.time = self.arrival_times[self.arrival_index]
      self.arrival_index += 1

      # this is the first block in a new cycle
      self.blocks_per_cycle.append(self.blocks_this_cycle)
      self.blocks_this_cycle = 1

      self.busy_period_begin = self.time
      self.consistent = False
    elif self.arrival_times[self.arrival_index] <= self.time + self.dt:
      arrival = True

      self.dt = self.arrival_times[self.arrival_index] - self.time
      self.time = self.arrival_times[self.arrival_index]
      self.arrival_index += 1
      self.blocks_this_cycle += 1
    else:
      self.time += self.dt

    # compute running system statistics (using the new dt and the old state)
    self.__compute_running_stats(arrival)

    # arrival
    if arrival:
      new_block_source = random.randint(self.N)
      self.block_array = np.append(self.block_array, np.zeros((1, self.N)), 0)
      self.block_array[-1, new_block_source] = 1
      self.block_sources = np.append(self.block_sources, np.zeros((1, self.N)), 0)
      self.block_sources[-1, new_block_source] = 1
      self.block_times = np.append(self.block_times, self.time)
      return
    # transmission
    else:
      sending_peer = random.randint(self.N)
      receiving_peer = random.randint(self.N - 1)
      if receiving_peer >= sending_peer:
        receiving_peer += 1

      # get the indices of sendable blocks
      sendable_blocks = [index for index in range(self.block_array.shape[0]) if \
                        self.block_array[index, sending_peer] == 1 and \
                        self.block_array[index, receiving_peer] == 0]
      #print(sending_peer)
      #print(receiving_peer)
      #print(sendable_blocks)

      # do nothing if there are no blocks to send (in general)
      if len(sendable_blocks) == 0:
        return
      else:
        # get the block index to send
        block_to_send = self.__get_block_to_send(sendable_blocks, sending_peer)
        # do nothing if the policy does not send a block
        if block_to_send == None:
          return
        # send the block
        else:
          self.block_array[block_to_send, receiving_peer] = 1
          # check for consistent state; set the flag if need be
          if np.all(self.block_array == 1):
            self.consistent = True
            self.cycles += 1
          return


  # remove all blocks that have completely propagated from the state matrices
  def __remove_inactive_blocks(self):
    if self.consistent:
      self.block_array = np.zeros((0, self.N))
      self.block_sources = np.zeros((0, self.N))
    else:
      index = 0
      while index < self.block_array.shape[0]:
        if np.all(self.block_array[index, :] == 1):
          self.block_array = np.delete(self.block_array, index, 0)
          self.block_sources = np.delete(self.block_sources, index, 0)
          self.block_times = np.delete(self.block_times, index, 0)
        else:
          index += 1


  # get the index of the block to send (depending on the policy)
  #   causal: send the oldest block which can be sent
  #   selfish: send the oldest block for which the sending peer is the source
  #   hybrid: send the oldest block for which the sending peer is the source.
  #           if no such block exists, send the oldest block which can be sent
  def __get_block_to_send(self, sendable_blocks, sending_peer):
    if self.policy == 'oldest-first':
      return sendable_blocks[0]
    elif self.policy == 'random':
      idx = random.randint(len(sendable_blocks))
      return sendable_blocks[idx]
    else: # self.policy == 'opportunistic'
      sending_peer_source_blocks = [block for block in sendable_blocks if \
                                    self.block_sources[block, sending_peer] == 1]
      if len(sending_peer_source_blocks) > 0:
        return sending_peer_source_blocks[0]
      else: 
        # if there are no source blocks to send
        return sendable_blocks[0]


  # compute the running statistics
  def __compute_running_stats(self, arrival):
    self.num_active_blocks_integral += self.dt * self.block_array.shape[0]
    if arrival:
      self.num_active_blocks_list.append(self.block_array.shape[0])
    self.num_missing_block_copies_integral += self.dt * \
                                  (self.block_array.shape[0] * self.block_array.shape[1] - np.sum(self.block_array))
    if len(self.block_times) > 0:
      self.age_of_information += self.dt * (self.time - self.block_times[0])
      # if not, we are consistent so we can add zero to the running tally
 
  # compute the final statistics 
  def compute_final_stats(self):
    print('\rMean number of active blocks: ' + str(self.num_active_blocks_integral/self.time))
    print('Mean number of missing block copies: ' + str(self.num_missing_block_copies_integral/self.time))
    print('Mean cycle length: ' + str(np.mean(self.cycle_lengths)))
    print('Mean blocks per cycle: ' + str(np.mean(self.blocks_per_cycle)))
    print('Mean Age of Information: ' + str(self.age_of_information/self.time))
