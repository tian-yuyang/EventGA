"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: path_finder.py
Description: Implements various path finding functions for generative agents.
Some of the functions are defunct. 
"""
import numpy as np
from maze_new import Maze

def print_maze(maze):
  for row in maze:
    for item in row:
      print(item, end='')
    print()


def path_finder_v2(a, start, end, collision_block_char, verbose=False):
  def make_step(m, k):
    for i in range(len(m)):
      for j in range(len(m[i])):
        if m[i][j] == k:
          if i>0 and m[i-1][j] == 0 and a[i-1][j] == 0:
            m[i-1][j] = k + 1
          if j>0 and m[i][j-1] == 0 and a[i][j-1] == 0:
            m[i][j-1] = k + 1
          if i<len(m)-1 and m[i+1][j] == 0 and a[i+1][j] == 0:
            m[i+1][j] = k + 1
          if j<len(m[i])-1 and m[i][j+1] == 0 and a[i][j+1] == 0:
             m[i][j+1] = k + 1

  new_maze = []
  for row in a: 
    new_row = []
    for j in row:
      if j == collision_block_char: 
        new_row += [1]
      else: 
        new_row += [0]
    new_maze += [new_row]
  a = new_maze

  m = []
  for i in range(len(a)):
      m.append([])
      for j in range(len(a[i])):
          m[-1].append(0)
  i,j = start
  m[i][j] = 1 

  k = 0
  except_handle = 200
  while m[end[0]][end[1]] == 0:
      k += 1
      make_step(m, k)

      if except_handle == 0: 
        break
      except_handle -= 1 

  i, j = end
  k = m[i][j]
  the_path = [(i,j)]
  while k > 1:
    if i > 0 and m[i - 1][j] == k-1:
      i, j = i-1, j
      the_path.append((i, j))
      k-=1
    elif j > 0 and m[i][j - 1] == k-1:
      i, j = i, j-1
      the_path.append((i, j))
      k-=1
    elif i < len(m) - 1 and m[i + 1][j] == k-1:
      i, j = i+1, j
      the_path.append((i, j))
      k-=1
    elif j < len(m[i]) - 1 and m[i][j + 1] == k-1:
      i, j = i, j+1
      the_path.append((i, j))
      k -= 1
        
  the_path.reverse()
  return the_path


def path_finder(maze, start, end, collision_block_char, verbose=False):
  # EMERGENCY PATCH
  start = (start[1], start[0])
  end = (end[1], end[0])
  # END EMERGENCY PATCH

  path = path_finder_v2(maze, start, end, collision_block_char, verbose)

  new_path = []
  for i in path: 
    new_path += [(i[1], i[0])]
  path = new_path
  
  return path


def closest_coordinate(curr_coordinate, target_coordinates): 
  min_dist = None
  closest_coordinate = None
  for coordinate in target_coordinates: 
    a = np.array(coordinate)
    b = np.array(curr_coordinate)
    dist = abs(np.linalg.norm(a-b))
    if not closest_coordinate: 
      min_dist = dist
      closest_coordinate = coordinate
    else: 
      if min_dist > dist: 
        min_dist = dist
        closest_coordinate = coordinate

  return closest_coordinate


def path_finder_2(maze, start, end, collision_block_char, verbose=False):
  # start => persona_a
  # end => persona_b
  start = list(start)
  end = list(end)

  t_top = (end[0], end[1]+1)
  t_bottom = (end[0], end[1]-1)
  t_left = (end[0]-1, end[1])
  t_right = (end[0]+1, end[1])
  pot_target_coordinates = [t_top, t_bottom, t_left, t_right]

  maze_width = len(maze[0]) 
  maze_height = len(maze)
  target_coordinates = []
  for coordinate in pot_target_coordinates: 
    if coordinate[0] >= 0 and coordinate[0] < maze_width and coordinate[1] >= 0 and coordinate[1] < maze_height: 
      target_coordinates += [coordinate]

  target_coordinate = closest_coordinate(start, target_coordinates)

  path = path_finder(maze, start, target_coordinate, collision_block_char, verbose=False)
  return path


def path_finder_3(maze, start, end, collision_block_char, verbose=False):
  # start => persona_a
  # end => persona_b

  curr_path = path_finder(maze, start, end, collision_block_char, verbose=False)
  if len(curr_path) <= 2: 
    return []
  else: 
    a_path = curr_path[:int(len(curr_path)/2)]
    b_path = curr_path[int(len(curr_path)/2)-1:]
  b_path.reverse()

  print (a_path)
  print (b_path)
  return a_path, b_path


if __name__ == '__main__':
  maze = Maze()
  
  print(path_finder(maze.collision_maze, (78, 14), (75, 65), 1, verbose=True))




