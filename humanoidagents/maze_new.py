"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: maze.py
Description: Defines the Maze class, which represents the map of the simulated
world in a 2-dimensional matrix. 
"""
import json
import numpy
import datetime
import pickle
import time
import math
import csv
import os

from utils import *


env_matrix = "../miniAgent_sampleMap_new"

def read_file_to_list(curr_file, header=False, strip_trail=True): 
  """
  Reads in a csv file to a list of list. If header is True, it returns a 
  tuple with (header row, all rows)
  ARGS:
    curr_file: path to the current csv file. 
  RETURNS: 
    List of list where the component lists are the rows of the file. 
  """
  if not header: 
    analysis_list = []
    with open(curr_file) as f_analysis_file: 
      data_reader = csv.reader(f_analysis_file)
      for count, row in enumerate(data_reader): 
        if strip_trail: 
          row = [i.strip() for i in row]
        analysis_list += [row]
    return analysis_list
  else: 
    analysis_list = []
    with open(curr_file) as f_analysis_file: 
      data_reader = csv.reader(f_analysis_file)
      for count, row in enumerate(data_reader): 
        if strip_trail: 
          row = [i.strip() for i in row]
        analysis_list += [row]
    return analysis_list[0], analysis_list[1:]

class Maze: 
  def __init__(self): 
    # Reading in the meta information about the world. If you want tp see the
    # example variables, check out the maze_meta_info.json file. 
    meta_info = json.load(open(f"{env_matrix}/map/maze_meta_info.json"))
    # <maze_width> and <maze_height> denote the number of tiles make up the 
    # height and width of the map. 
    self.maze_width = int(meta_info["maze_width"])
    self.maze_height = int(meta_info["maze_height"])
    # <sq_tile_size> denotes the pixel height/width of a tile. 
    self.sq_tile_size = int(meta_info["sq_tile_size"])
    # <special_constraint> is a string description of any relevant special 
    # constraints the world might have. 
    # e.g., "planning to stay at home all day and never go out of her home"
    self.special_constraint = meta_info["special_constraint"]

    # READING IN SPECIAL BLOCKS
    # Special blocks are those that are colored in the Tiled map. 

    # Here is an example row for the arena block file: 
    # e.g., "25335, Double Studio, Studio, Common Room"
    # And here is another example row for the game object block file: 
    # e.g, "25331, Double Studio, Studio, Bedroom 2, Painting"

    # Notice that the first element here is the color marker digit from the 
    # Tiled export. Then we basically have the block path: 
    # World, Sector, Arena, Game Object -- again, these paths need to be 
    # unique within an instance of Reverie. 
    blocks_folder = f"{env_matrix}/map/special_blocks"
   
    _sb = blocks_folder + "/sector_blocks.csv"
    sb_rows = read_file_to_list(_sb, header=False)
    sb_dict = dict()
    for i in sb_rows: sb_dict[i[0]] = i[-1]
    
    _ab = blocks_folder + "/arena_blocks.csv"
    ab_rows = read_file_to_list(_ab, header=False)
    ab_dict = dict()
    for i in ab_rows: ab_dict[i[0]] = i[-1]
    
    # _gob = blocks_folder + "/game_object_blocks.csv"
    # self.gob_rows = read_file_to_list(_gob, header=False)
    # gob_dict = dict()
    self.gob_feature_dict = dict()
    self.object_list =set()
    self.feature_dict = dict()
    # for i in self.gob_rows:
    #   gob_dict[i[0]] = i[1]
    #   gob_feature_dict[i[0]] = {}
    #   features = [i[4],','.join(i[5:-1]),i[-1]]
    #   for item in features:
    #     key, value = item.split('=')
    #     key = key.strip()
    #     value = value.strip()
    #     gob_feature_dict[i[0]][key] = eval(value)
    
    
    map_data_folder = f"{env_matrix}/map/map_data"
    objects_folder = f"{env_matrix}/objects"
    gb_data = json.load(open(f"{map_data_folder}/objects_data.json"))["objects"]
    buildings_folder = f"{env_matrix}/buildings"
    bd_data = json.load(open(f"{map_data_folder}/buildings_data.json"))["buildings"]
    
    objects_info = {}
    for subdir in os.listdir(objects_folder):
      subdir_path = os.path.join(objects_folder, subdir)
      info = json.load(open(f"{subdir_path}/maze_meta_info.json"))
      objects_info[info['typeId']] = [info['maze_width'], info['maze_height']]
      
    buildings_info = {}
    for subdir in os.listdir(buildings_folder):
      subdir_path = os.path.join(buildings_folder, subdir)
      info = json.load(open(f"{subdir_path}/maze_meta_info.json"))
      buildings_info[info['typeId']] = [info['maze_width'], info['maze_height']]
    
    
    # for i in gob_rows: 
    #   item = set([a.strip() for a in i[-1][1:-1].split(',')])
    #   iob_dict[i[0]] = item
    #   self.item_set.update(item)

    # [SECTION 3] Reading in the matrices 
    # This is your typical two dimensional matrices. It's made up of 0s and 
    # the number that represents the color block from the blocks folder. 
    maze_folder = f"{env_matrix}/map/maze"

    _cm = maze_folder + "/collision_maze.csv"
    self.collision_maze = read_file_to_list(_cm, header=False)
    _sm = maze_folder + "/sector_maze.csv"
    sector_maze = read_file_to_list(_sm, header=False)
    
    _am = maze_folder + "/arena_maze.csv"
    arena_maze = read_file_to_list(_am, header=False)
    
    self.occupied_tiles = self.collision_maze.copy()
    for i in range(self.maze_height):
      for j in range(self.maze_width):
        if self.collision_maze[i][j] != "0":
          self.occupied_tiles[i][j] = 1
        else:
          self.occupied_tiles[i][j] = 0
    

    # Loading the maze. The mazes are taken directly from the json exports of
    # Tiled maps. They should be in csv format. 
    # Importantly, they are "not" in a 2-d matrix format -- they are single 
    # row matrices with the length of width x height of the maze. So we need
    # to convert here. 
    # We can do this all at once since the dimension of all these matrices are
    # identical (e.g., 70 x 40).
    # example format: [['0', '0', ... '25309', '0',...], ['0',...]...]
    # 25309 is the collision bar number right now.
    game_object_maze = []
    for i in range(0, self.maze_height):
      game_object_maze += [['' for j in range(self.maze_width)]]
    uniqueid_to_name = {}
    for object in gb_data:
      uniqueid_to_name[object["uniqueId"]] = object["itemName"]
    for object in gb_data:
      if object["x"] != -1 and object["y"] != -1:
        # print(object["itemName"])
        # game_object_maze[object['x']][object['y']] = [object['itemName'], object['uniqueId']]
        self.gob_feature_dict[object['uniqueId']] = {'position': []}
        
        for x in range(objects_info[object['typeId']][0]):
          for y in range(objects_info[object['typeId']][1]):
              game_object_maze[object['y']+y][object['x']+x] = [object['itemName'], object['uniqueId']]
              self.gob_feature_dict[object['uniqueId']]['position'] += [(object['x']+x, object['y']+y)]
        
        for item in object["attributes"]:
          key, value = item["key"], item["value"]
          if key == 'quantity':
            self.gob_feature_dict[object["uniqueId"]]['quantity'] = eval(value)
          if key == 'container':
            containing = value.split(',')
            self.gob_feature_dict[object["uniqueId"]]['container'] = []
            for item in containing:
              item = uniqueid_to_name[item.strip()]
              self.gob_feature_dict[object["uniqueId"]]['container'].append(item)
          
    
    
    self.tiles = []
    self.spatial_structure = dict()
    # self.item_list = dict()
    for i in range(self.maze_height): 
      row = []
      for j in range(self.maze_width):
        tile_details = dict()
        
        tile_details["sector"] = ""
        if sector_maze[i][j] in sb_dict: 
          for item in bd_data:
            if item['y'] <= i < item['y'] + buildings_info[item['typeId']][1] and \
               item['x'] <= j < item['x'] + buildings_info[item['typeId']][0]:
              tile_details["sector"] = item['itemName']
          # tile_details["sector"] = sb_dict[sector_maze[i][j]]
          if tile_details["sector"] not in self.spatial_structure:
            self.spatial_structure[tile_details["sector"]] = {}
        
        tile_details["arena"] = ""
        if arena_maze[i][j] in ab_dict: 
          tile_details["arena"] = ab_dict[arena_maze[i][j]]
          try:
            if tile_details["arena"] not in self.spatial_structure[tile_details["sector"]]:
              self.spatial_structure[tile_details["sector"]][tile_details["arena"]] = []
          except:
            print(i,j)
        
        tile_details["game_object"] = ""
        if game_object_maze[i][j]: 
          tile_details["game_object"] = game_object_maze[i][j][0]
          tile_details["uniqueid"] = game_object_maze[i][j][1]
          try:
            if tile_details["arena"] in self.spatial_structure[tile_details["sector"]]:
              self.spatial_structure[tile_details["sector"]][tile_details["arena"]] += [tile_details["game_object"]]
          except:
            print(i,j)

        
        tile_details["collision"] = False
        if self.collision_maze[i][j] != "0": 
          tile_details["collision"] = True

        tile_details["events"] = set()
        
        row += [tile_details]
      self.tiles += [row]
      
    self.address_tiles = dict()
    for i in range(self.maze_height):
      for j in range(self.maze_width): 
        addresses = []
        if self.tiles[i][j]["sector"]: 
          add = f'{self.tiles[i][j]["sector"]}'
          addresses += [add]
        if self.tiles[i][j]["arena"]: 
          add = f'{self.tiles[i][j]["sector"]}:'
          add += f'{self.tiles[i][j]["arena"]}'
          addresses += [add]
        if self.tiles[i][j]["game_object"]: 
          add = f'{self.tiles[i][j]["sector"]}:'
          add += f'{self.tiles[i][j]["arena"]}:'
          add += f'{self.tiles[i][j]["game_object"]}'
          addresses += [add]

        for add in addresses: 
          if add in self.address_tiles: 
            self.address_tiles[add].add((j, i))
          else: 
            self.address_tiles[add] = set([(j, i)])
        
        if self.tiles[i][j]["game_object"]:
          add = f'{self.tiles[i][j]["sector"]}:'
          add += f'{self.tiles[i][j]["arena"]}:'
          add += f'{self.tiles[i][j]["game_object"]}'
          # self.object_list.add(add)
          self.feature_dict[add] = self.gob_feature_dict[self.tiles[i][j]['uniqueid']]
          if 'container' in self.gob_feature_dict[self.tiles[i][j]['uniqueid']] and \
          self.gob_feature_dict[self.tiles[i][j]['uniqueid']]['container'] is not None:
            for item in self.gob_feature_dict[self.tiles[i][j]['uniqueid']]['container']:
              # feature_ = gob_feature_dict[f"{item}"]
              self.object_list.add(f"{item}")
              # self.feature_dict[f"{item}"] = feature_
              if item not in self.feature_dict:
                self.feature_dict[f"{item}"] = set()
              self.feature_dict[f"{item}"].add(add)


  def turn_coordinate_to_tile(self, px_coordinate): 
    """
    Turns a pixel coordinate to a tile coordinate. 

    INPUT
      px_coordinate: The pixel coordinate of our interest. Comes in the x, y
                     format. 
    OUTPUT
      tile coordinate (x, y): The tile coordinate that corresponds to the 
                              pixel coordinate. 
    EXAMPLE OUTPUT 
      Given (1600, 384), outputs (50, 12)
    """
    x = math.ceil(px_coordinate[0]/self.sq_tile_size)
    y = math.ceil(px_coordinate[1]/self.sq_tile_size)
    return (x, y)


  def access_tile(self, tile): 
    """
    Returns the tiles details dictionary that is stored in self.tiles of the 
    designated x, y location. 

    INPUT
      tile: The tile coordinate of our interest in (x, y) form.
    OUTPUT
      The tile detail dictionary for the designated tile. 
    EXAMPLE OUTPUT
      Given (58, 9), 
      self.tiles[9][58] = {'world': 'double studio', 
            'sector': 'double studio', 'arena': 'bedroom 2', 
            'game_object': 'bed', 'spawning_location': 'bedroom-2-a', 
            'collision': False,
            'events': {('double studio:double studio:bedroom 2:bed',
                       None, None)}} 
    """
    x = tile[0]
    y = tile[1]
    return self.tiles[y][x]


  def get_tile_path(self, tile, level): 
    """
    Get the tile string address given its coordinate. You designate the level
    by giving it a string level description. 

    INPUT: 
      tile: The tile coordinate of our interest in (x, y) form.
      level: world, sector, arena, or game object
    OUTPUT
      The string address for the tile.
    EXAMPLE OUTPUT
      Given tile=(58, 9), and level=arena,
      "double studio:double studio:bedroom 2"
    """
    x = tile[0]
    y = tile[1]
    tile = self.tiles[y][x]

    path = f"{tile['sector']}"
    if level == "sector": 
      return path
    else: 
      path += f":{tile['arena']}"

    if level == "arena": 
      return path
    else: 
      path += f":{tile['game_object']}"

    return path


  def get_nearby_tiles(self, tile, vision_r): 
    """
    Given the current tile and vision_r, return a list of tiles that are 
    within the radius. Note that this implementation looks at a square 
    boundary when determining what is within the radius. 
    i.e., for vision_r, returns x's. 
    x x x x x 
    x x x x x
    x x P x x 
    x x x x x
    x x x x x

    INPUT: 
      tile: The tile coordinate of our interest in (x, y) form.
      vision_r: The radius of the persona's vision. 
    OUTPUT: 
      nearby_tiles: a list of tiles that are within the radius. 
    """
    left_end = 0
    if tile[0] - vision_r > left_end: 
      left_end = tile[0] - vision_r

    right_end = self.maze_width - 1
    if tile[0] + vision_r + 1 < right_end: 
      right_end = tile[0] + vision_r + 1

    bottom_end = self.maze_height - 1
    if tile[1] + vision_r + 1 < bottom_end: 
      bottom_end = tile[1] + vision_r + 1

    top_end = 0
    if tile[1] - vision_r > top_end: 
      top_end = tile[1] - vision_r 

    nearby_tiles = []
    for i in range(left_end, right_end): 
      for j in range(top_end, bottom_end): 
        nearby_tiles += [(i, j)]
    return nearby_tiles


  def add_event_from_tile(self, curr_event, time, tile): 
    """
    Add an event triple to a tile.  

    INPUT: 
      curr_event: Current event triple. 
        e.g., ('double studio:double studio:bedroom 2:bed', None,
                None)
      tile: The tile coordinate of our interest in (x, y) form.
    OUPUT: 
      None
    """
    self.tiles[tile[1]][tile[0]]["events"].add((curr_event, time))


  def remove_event_from_tile(self, curr_event, tile):
    """
    Remove an event triple from a tile.  

    INPUT: 
      curr_event: Current event triple. 
        e.g., ('double studio:double studio:bedroom 2:bed', None,
                None)
      tile: The tile coordinate of our interest in (x, y) form.
    OUPUT: 
      None
    """
    curr_tile_ev_cp = self.tiles[tile[1]][tile[0]]["events"].copy()
    for event in curr_tile_ev_cp: 
      if event == curr_event:  
        self.tiles[tile[1]][tile[0]]["events"].remove(event)


  def turn_event_from_tile_idle(self, curr_event, tile):
    curr_tile_ev_cp = self.tiles[tile[1]][tile[0]]["events"].copy()
    for event in curr_tile_ev_cp: 
      if event == curr_event:  
        self.tiles[tile[1]][tile[0]]["events"].remove(event)
        new_event = (event[0], None, None, None)
        self.tiles[tile[1]][tile[0]]["events"].add(new_event)


  def remove_subject_events_from_tile(self, subject, tile):
    """
    Remove an event triple that has the input subject from a tile. 

    INPUT: 
      subject: "Isabella Rodriguez"
      tile: The tile coordinate of our interest in (x, y) form.
    OUPUT: 
      None
    """
    curr_tile_ev_cp = self.tiles[tile[1]][tile[0]]["events"].copy()
    for event in curr_tile_ev_cp: 
      if event[0] == subject:  
        self.tiles[tile[1]][tile[0]]["events"].remove(event)


  def add_item_from_tile(self, address, item): 
    """
    Add an item to a tile.  

    INPUT: 
      item: The item string. 
        e.g., 'bag'
      tile: The tile coordinate of our interest in (x, y) form.
    OUPUT: 
      None
    """
    potential_tiles = self.address_tiles[address]
    for tile in potential_tiles:
      if item in self.tiles[tile[1]][tile[0]]["item"]:
        continue
      self.tiles[tile[1]][tile[0]]["item"].add(item)
      return
    
  def remove_item_from_tile(self, tile, item): 
    """
    Remove an item from a tile.  

    INPUT: 
      tile: The tile coordinate of our interest in (x, y) form.
    OUPUT: 
      None
    """
    if item in self.tiles[tile[1]][tile[0]]["item"]:
      self.tiles[tile[1]][tile[0]]["item"].remove(item)
  

  def access_item_from_object(self, object):
    return self.item_list[object]


  def generate_distance_matrix(self):
    self.distance_matrix = {}
    for i in range(self.maze_height):
      for j in range(self.maze_width):
        if self.tiles[i][j]["sector"] != "":
          place_1 = ":".join([self.tiles[i][j]["world"], 
                                  self.tiles[i][j]["sector"]])
          for a in range(self.maze_height):
            for b in range(self.maze_width):
              if self.tiles[a][b]["sector"] != "":
                place_2 = ":".join([self.tiles[a][b]["world"], 
                                  self.tiles[a][b]["sector"]])
                dist = abs(i-a) + abs(j-b)
                if (place_1, place_2) not in self.distance_matrix:
                  self.distance_matrix[(place_1, place_2)] = (dist+9) // 10
                else:
                  self.distance_matrix[(place_1, place_2)] = min(self.distance_matrix[(place_1, place_2)],(dist+9)//10)


  def get_str_accessible_sectors(self):
    x = ", ".join(list(self.spatial_structure.keys()))
    return x

  def get_str_accessible_sector_arenas(self, sector):
    if not sector in self.spatial_structure:
        return ""
    x = ", ".join(list(self.spatial_structure[sector].keys()))
    return x

  def get_str_accessible_sector_arena_game_objects(self, sector, arena):
    if not sector in self.spatial_structure:
        return ""
    if not arena in self.spatial_structure[sector]:
        return ""

    try:
        x = ", ".join(list(self.spatial_structure[sector][arena]))
    except Exception as e:
        x = ", ".join(list(self.spatial_structure[sector][arena.lower()]))
    return x

  def get_accessible_sectors(self):
    return list(self.spatial_structure.keys())

  def get_accessible_sector_arenas(self, sector):
    if not sector in self.spatial_structure:
        return []
    return list(self.spatial_structure[sector].keys())

  def get_accessible_sector_arena_game_objects(self, sector, arena):
    if not sector in self.spatial_structure:
        return []
    if not arena in self.spatial_structure[sector]:
        return []

    return self.spatial_structure[sector][arena]
  
  def assign_location(self, tile):
    if self.occupied_tiles[tile[1]][tile[0]] == 1:
      dx = [-1, 0, 1, 0]
      dy = [0, -1, 0, 1]
      for i in range(4):
        new_tile = (tile[0] + dx[i], tile[1] + dy[i])
        if new_tile[0] < 0 or new_tile[0] >= self.maze_width or \
           new_tile[1] < 0 or new_tile[1] >= self.maze_height:
          continue
        if self.occupied_tiles[new_tile[1]][new_tile[0]] == 0:
          self.occupied_tiles[new_tile[1]][new_tile[0]] = 1
          return new_tile
      
      for i in range(4):
        for j in range(4):
          new_tile = (tile[0] + dx[i] + dx[j], tile[1] + dy[i] + dy[j])
          if new_tile[0] < 0 or new_tile[0] >= self.maze_width or \
             new_tile[1] < 0 or new_tile[1] >= self.maze_height:
            continue
          if self.occupied_tiles[new_tile[1]][new_tile[0]] == 0:
            self.occupied_tiles[new_tile[1]][new_tile[0]] = 1
            return new_tile
      return tile
    else:
      self.occupied_tiles[tile[1]][tile[0]] = 1
      return tile
  
  def release_location(self, tile):
    self.occupied_tiles[tile[1]][tile[0]] = 0
  
  

if __name__ == "__main__":
  # Example usage of the Maze class
  maze = Maze()
  # print(maze.address_tiles)
  # print(maze.address_tiles["Dorm for Oak Hill College:Ayesha Khan's room:desk"])
  for sector in maze.get_accessible_sectors():
    for arena in maze.get_accessible_sector_arenas(sector):
      if len(maze.get_accessible_sector_arena_game_objects(sector, arena)) == 0:
        print(sector, arena)
            