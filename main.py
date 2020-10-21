import numpy as np
import multiprocessing as mp

from datetime import datetime

from block_propagation import Block_Propagation_Simulator

def main():
  # run(N, policy, num_cycles, arrival_rate, num_processes)
  run(100, 'causal', 30000, 0.5, 5)
  run(100, 'hybrid', 30000, 0.5, 5)
  #run(100, 'selfish', 30000, 0.5, 5)

  run(100, 'causal', 30000, 0.7, 5)
  run(100, 'hybrid', 30000, 0.7, 5)
  #run(100, 'selfish', 30000, 0.7, 5)

  run(100, 'causal', 30000, 0.9, 5)
  run(100, 'hybrid', 30000, 0.9, 5)
  #run(100, 'selfish', 30000, 0.9, 5)

  run(100, 'causal', 30000, 0.95, 5)
  run(100, 'hybrid', 30000, 0.95, 5)
  #run(100, 'selfish', 30000, 0.95, 5)

def run(N, policy, num_blocks, arrival_rate, num_processes):
  output = mp.Queue()
  processes = [mp.Process(target=run_process, \
               args=(output, N, policy, num_blocks, arrival_rate)) \
               for x in range(num_processes)]

  start_time = datetime.now()
  for p in processes:
    p.start()

  for p in processes:
    p.join()

  results = [output.get() for p in processes]

  total_time = np.sum([results[p][0] for p in range(num_processes)])

  num_active_blocks_integral = np.sum([results[p][1] for p in range(num_processes)])
  mean_active_blocks = num_active_blocks_integral/total_time

  num_missing_block_copies_integral = np.sum([results[p][2] for p in range(num_processes)])
  mean_missing_block_copies = num_missing_block_copies_integral/total_time

  mean_cycle_length = np.mean([results[p][3] for p in range(num_processes)])

  mean_blocks_per_cycle = np.mean([results[p][4] for p in range(num_processes)])

  print('N: ' + str(N))
  print('Policy: ' + policy)
  print('Number of Blocks: ' + str(num_processes*num_blocks))
  print('Arrival Rate: ' + str(arrival_rate))

  print('Mean Number of Blocks in System: ' + str(mean_active_blocks))
  print('Mean Number of Missing Block Copies: ' + str(mean_missing_block_copies))
  #print('Mean Cycle Length: ' + str(mean_cycle_length))
  #print('Mean Number of Blocks per Cycle: ' + str(mean_blocks_per_cycle))
  
  end_time = datetime.now()
  print('Time Elapsed: ' + str(end_time - start_time))

def run_process(output, N, policy, num_blocks, arrival_rate):
  simulator = Block_Propagation_Simulator(N, policy, num_blocks, arrival_rate)
  output.put(simulator.run())

if __name__=='__main__':
  main()
