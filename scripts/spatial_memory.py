import json

class MemoryTree:
    def __init__(self, f_saved):
        self.tree = json.load(open(f_saved))

    def print_tree(self):
        def _print_tree(tree, depth):
            dash = " >" * depth
            if type(tree) == type(list()):
                if tree:
                    print(dash, tree)
                return

            for key, val in tree.items():
                if key:
                    print(dash, key)
                _print_tree(val, depth + 1)

        _print_tree(self.tree, 0)

    def save(self, out_json):
        with open(out_json, "w") as outfile:
            json.dump(self.tree, outfile)

    def get_str_accessible_sectors(self):
        """
    Returns a summary string of all the arenas that the persona can access 
    within the current sector. 

    Note that there are places a given persona cannot enter. This information
    is provided in the persona sheet. We account for this in this function. 

    INPUT
      None
    OUTPUT 
      A summary string of all the arenas that the persona can access. 
    EXAMPLE STR OUTPUT
      "bedroom, kitchen, dining room, office, bathroom"
    """
        x = ", ".join(list(self.tree.keys()))
        return x

    def get_str_accessible_sector_arenas(self, sector):
        """
    Returns a summary string of all the arenas that the persona can access 
    within the current sector. 

    Note that there are places a given persona cannot enter. This information
    is provided in the persona sheet. We account for this in this function. 

    INPUT
      None
    OUTPUT 
      A summary string of all the arenas that the persona can access. 
    EXAMPLE STR OUTPUT
      "bedroom, kitchen, dining room, office, bathroom"
    """
        if not sector in self.tree:
            return ""
        x = ", ".join(list(self.tree[sector].keys()))
        return x

    def get_str_accessible_arena_game_objects(self, arena):
        """
    Get a str list of all accessible game objects that are in the arena. If 
    temp_address is specified, we return the objects that are available in
    that arena, and if not, we return the objects that are in the arena our
    persona is currently in. 

    INPUT
      temp_address: optional arena address
    OUTPUT 
      str list of all accessible game objects in the gmae arena. 
    EXAMPLE STR OUTPUT
      "phone, charger, bed, nightstand"
    """
        curr_sector, curr_arena = arena.split(":")

        if not curr_arena:
            return ""

        try:
            x = ", ".join(list(self.tree[curr_sector][curr_arena]))
        except Exception as e:
            x = ", ".join(list(self.tree[curr_sector][curr_arena.lower()]))
        return x
      
    def get_all_accessible_game_objects(self):
        known_game_objects = []
        for world in self.tree:
            for sector in self.tree[world]:
                for arena in self.tree[world][sector]:
                    for game_object in self.tree[world][sector][arena]:
                        known_game_objects += [f"{world}:{sector}:{arena}:{game_object}"]
        return known_game_objects
