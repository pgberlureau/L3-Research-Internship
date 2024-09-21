import numpy as np
import networkx as nx
import geopandas as gpd
import shapely
import xgi
from geopandas import GeoSeries
from itertools import combinations
from scipy.cluster.hierarchy import DisjointSet
from mapclassify import greedy
from shapely.geometry import MultiLineString, LineString
from shapely import STRtree, intersection, line_merge

def preprocess(gdf, buffsize):
    """
    A function to get rid of multiple linestrings on the "same" street by 
    computing the skeleton of the union of all buffers around LineStrings using Voronoi polygons.
    
    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        Input shapefile

    buffsize:
        Size of buffers used to merge linestrings        
        
    Returns
    -------
    gpd.GeoDataFrame
        Preprocessed GeoDataFrame Geometry column
    """
    def construct_centerline(input_geometry, interpolation_distance=buffsize/10):
       #find the voronoi verticies (equivalent to Centerline._get_voronoi_vertices_and_ridges())
       borders = input_geometry.segmentize(interpolation_distance) #To have smaler verticies (equivalent to Centerline._get_densified_borders())
       voronoied = shapely.voronoi_polygons(borders,only_edges=True) #equivalent to the scipy.spatial.Voronoi
    
       #to select only the linestring within the input geometry (equivalent to Centerline._linestring_is_within_input_geometry)
       centerlines = gpd.GeoDataFrame(geometry=gpd.GeoSeries(voronoied.geoms)).sjoin(gpd.GeoDataFrame(geometry=gpd.GeoSeries(input_geometry)),predicate="within")
    
       return centerlines.unary_union

    buffer = gdf.buffer(buffsize) #Creates the buffer
    union = gpd.GeoSeries(list(shapely.union_all(buffer).geoms), crs=buffer.crs).buffer(buffsize*(1/10-1)) #Computes the union of all buffers and shrink it
    return buffer, union, gpd.GeoDataFrame(geometry=np.hstack(np.vectorize(lambda x: construct_centerline(x).geoms)(union)), crs=gdf.crs)
    
    
def remove_isolated(gdf):
    """
    A function to get rid of isolated linestrings

    Parameters
    ----------
    gdf: gpd.GeoDataFrame 
        Input shapefile

    Returns
    -------
    gpd.GeoDataFrame
        Filtered GeoDataFrame
    """
    query = STRtree(gdf['geometry']).query(gdf['geometry'], 'intersects')
    return gdf.iloc[np.unique(query[:,query[0] != query[1]].flatten())]

class WayGraph:
    """
    A WayGraph is the primal graph of the geometry attribute of a GeoDataFrame in which every row is a LineString (no MultiLineString).
    The geometry attribute of each row is considered as an edge between its two ends which are nodes of the WayGraph.
    
    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        The GeoDataFrame from which the WayGraph should be built.
    
    Attributes
    ----------
    graph: networkx.MultiGraph
        The MultiGraph built from the gdf parameter

    gdf: geopandas.GeoDataFrame
        The GeoDataFrame containing only the geometry attribute of the gdf parameter
    """
    def __init__(self, gdf):
        edges = gdf['geometry'].apply(lambda x: (x.coords[0], x.coords[-1], {'geometry': x}))
        self.graph = nx.MultiGraph()
        self.graph.add_edges_from(edges)
        self.gdf = gpd.GeoDataFrame(geometry=gdf['geometry'])
        
    def draw(self, ax, shape_view=True):
        """
        Plots the WayGraph in the specified axis.

        Parameters
        ----------
        ax: matplotlib.pyplot.Artist
            The axis on which the WayGraph should be drawn.

        shape_view: bool
            If true then use LineStrings to draw edges, else use the default drawing of networkx.
        """
        if shape_view:
            self.gdf.plot(greedy(self.gdf), cmap="Set3", ax=ax)
        else:
            nx.draw_networkx(self.graph, {n:n for n in self.graph.nodes(data=False)}, node_size=10, with_labels=False, ax=ax)

class WayHyperGraph:
    """
    A WayHyperGraph is the hypergraph built from a WayGraph wg using the angle parameter the following way:
        for each node n:
            while there exists two edges of n forming an angle of 180° +- angle then "pair" those which make the closest angle to 180° and repeat it with remaining edges
    
    Parameters
    ----------
    wg: WayGraph
        The WayGraph from which the WayHyperGraph has to be built.

    angle: float
        The threshold which should be used to compute the hyperedges of the WayHyperGraph.
        
    Attributes
    ----------
    hypergraph: xgi.Hypergraph
        The hypergraph built from the wg parameter.
    
    gdf: geopandas.GeoDataFrame
        The GeoDataFrame with the single attribute geometry in which LineStrings of hyperedges have been merged.
    """
    def __init__(self, wg, angle=np.pi/3): 
        def nodes_of_edgeset(s):#Return the nodes which have edges in the set of edges s
            return list({u for u,v,_ in s}.union({v for u,v,_ in s}))

        def ordered_edge(e):#Return the input edge in a specified way to make them unique
            return (min(e[0], e[1]), max(e[0], e[1]), e[2])

        def angle_between(node, e1, e2, wg): #Compute the angle e1 and e2 make on node in the WayGraph wg using complex numbers
            l1 = wg.graph[e1[0]][e1[1]][e1[2]]['geometry'].coords #coords of the LineString associated to e1
            l2 = wg.graph[e2[0]][e2[1]][e2[2]]['geometry'].coords #coords of the LineString associated to e2

            if l1[0] == node: #Finding which end of l1 is connected to the node
                s1 = complex(*l1[1])-complex(*node) 
            else:
                s1 = complex(*l1[-2])-complex(*node)

            if l2[0] == node:
                s2 = complex(*l2[1])-complex(*node)
            else:
                s2 = complex(*l2[-2])-complex(*node)

            if s1!=complex(0):
                return abs(np.angle(-s2/s1))    
            else:
                return 0

        #Using DisjointSet to build hyperedges, as hyperedges are like equivalence class for edges
        ways = DisjointSet(list(map(ordered_edge, wg.graph.edges(data=False, keys=True)))) #Initially each edge is alone in it's equivalence class

        for node in wg.graph.nodes(data=False):
            #We compute the angle between all pairs of edges satisfying the threshold
            angles = {x : angle_between(node, x[0], x[1], wg) for x in combinations(wg.graph.edges(node, keys=True), 2) if angle_between(node, x[0], x[1], wg) < angle}
            while len(angles) > 0:
                e1, e2 = min(angles, key=angles.get) #Extract edges realizing the closest angle to pi
                ways.merge(ordered_edge(e1), ordered_edge(e2)) #Merge their equivalence class
                angles = {x:k for x,k in angles.items() if x[0] != e1 and x[1] != e1 and x[0] != e2 and x[1] != e2} #Remove the edge couple from the dictionnary

        hyperedges = np.vectorize(lambda w: MultiLineString([wg.graph[v[0]][v[1]][v[2]]['geometry'] for v in w]))(ways.subsets()) 
        self.spatial_hyperedges = np.vectorize(line_merge)(hyperedges) #Merges of equivalence class geometries (might still contains MultiLineStrings)
        merged = np.hstack( np.vectorize(lambda x: x if isinstance(x, LineString) else np.vectorize(LineString)(x.geoms))(self.spatial_hyperedges) ) #Flatten LineStringed merges (contains only LineStrings)
        self.gdf = gpd.GeoDataFrame(geometry=GeoSeries(merged), crs=wg.gdf.crs) #GeoDataFrame with merged as geometry column

        hyperedges = dict(zip(range(len(ways.subsets())), map(nodes_of_edgeset, ways.subsets()))) #Dictionnary required to build Hypergraph from equivalence classes
        self.hypergraph = xgi.Hypergraph(hyperedges) #Hypergraph
        self.graph = wg.graph #Original graph
        self.linegraph = self.to_line_graph() #Linegraph of the hypergraph
        
    
    def connectivity(self, e):
        """
        Returns connectivity of hyperedge e
        
        Parameters
        ----------
        e : int
            ID of the hyperedge 
            
        Returns
        -------
        int
            connectivity of e
        """

        return np.sum([self.graph.degree(n) for n in self.hypergraph.edges.members(e)]) - 2 * self.hypergraph.edges.order.asdict()[e]

    def degree(self, e):
        """
        Returns degree of hyperedge e

        Parameters
        ----------
        e : int
            ID of the hyperedge
            
        Returns
        -------
        int
            Degree of e 
        """
        return len(self.hypergraph.edges.neighbors(e))

    def spacing(self, e):
        """
        Returns the spacing of hyperedge e
        
        Parameters
        ----------
        e : int
            ID of the hyperedge
            
        Returns
        -------
        float
            Spacing of e
        """
        return self.spatial_hyperedges[e].length / self.connectivity(e)
    
    def orthogonality(self, e):
        """
        Returns the orthogonality of hyperedge e
        
        Parameters
        ----------
        e : int
            ID of the hyperedge
            
        Returns
        -------
        float
            orthogonality of e
        """

        def angles_on_node(node): #Inner function which computes the sum of sin of the angle between adjcents edges to node and hyperedge e
            e_edges = [a for a in list(zip(*self.graph.edges(node, data='geometry')))[2] if self.spatial_hyperedges[e].covers(a)] #Edges of node which are in the hyperedge e

            def angle(e1, e2): #Inner^2 function which computes the sin of the angle between edge e1 and edge e2
                if node == e1.coords[0]:
                    s2 = complex(*e1.coords[1])-complex(*e1.coords[0])
                    
                else:
                    s2 = complex(*e1.coords[-2])-complex(*e1.coords[-1])
                    
                if node == e2.coords[0]:
                    s1 = complex(*e2.coords[1])-complex(*e2.coords[0])

                else:
                    s1 = complex(*e2.coords[-2])-complex(*e2.coords[-1])

                if s2 == 0:
                    return 0
                else:
                    return abs(np.sin(np.angle(s1 / s2)))

            return np.sum([np.min([angle(e1, e2) for e2 in e_edges]) for e1 in list(zip(*self.graph.edges(node, data='geometry')))[2]])

        return np.sum([angles_on_node(n) for n in self.hypergraph.edges.members(e)]) / self.connectivity(e)
    
    def to_line_graph(self, s=1, weights=None):
        """
        Correction de la fonction homonyme du package xgi, j'ai ouvert une issue github puis à leur demande une pull request qui a été accepté donc la fonction sera corrigée dans la prochaine release et pourra disparaître de ce code :)
        """
        if weights not in [None, "absolute", "normalized"]:
            raise xgi.exception.XGIError(
                f"{weights} not a valid weights option. Choices are "
                "None, 'absolute', and 'normalized'."
            )
        LG = nx.Graph()

        edge_label_dict = {tuple(edge): index for index, edge in self.hypergraph._edge.items()}

        LG.add_nodes_from(edge_label_dict.values())

        for edge1, edge2 in combinations(edge_label_dict.keys(), 2):
            # Check that the intersection size is larger than s
            intersection_size = len(set(edge1).intersection(set(edge2)))
            if intersection_size >= s:
                if not weights:
                    # Add unweighted edge
                    LG.add_edge(
                        edge_label_dict[edge1], edge_label_dict[edge2]
                    )
                else:
                    # Compute the (normalized) weight
                    weight = intersection_size
                    if weights == "normalized":
                        weight /= min([len(edge1), len(edge2)])
                    # Add edge with weight
                    LG.add_edge(
                        edge_label_dict[edge1],
                        edge_label_dict[edge2],
                        weight=weight,
                    )

        return LG 

    def closeness(self):
        """
        Calls networkx.closeness_centrality on the line graph of the hypergraph
        """
        return nx.closeness_centrality(self.linegraph)
    
    def betweenness(self):
        """
        Calls networkx.betweenness_centrality on the line graph of the hypergraph
        """

        return nx.betweenness_centrality(self.linegraph)
    
    def accessibility(self, e):
        """
        Computes accessibility of hyperedge e
        
        Parameters
        ----------
        e: int
            ID of the hyperedge
            
        Returns
        -------
        float:
            accessibility of e
        """
        def distance(v):
            try:
                res = nx.shortest_path_length(g, v, e)
            except:
                res = 0
            return res

        g = self.linegraph
        return np.vectorize(distance)(g.nodes) @ np.vectorize(shapely.length)(self.spatial_hyperedges)
    
    def draw(self, ax, shape_view=True):
        """
        Plots the WayHyperGraph in the specified axis
            
        Parameters
        ----------
        ax: matplotlib.pyplot.Artist
            The axis on which the WayHyperGraph should be drawn. 

        shape_view: bool
            If true then use LineStrings to draw edges, else use the default drawing of xgi
        """
        if shape_view:
            self.gdf.plot(greedy(self.gdf), cmap='Set3', ax=ax)
        else:
            xgi.draw(self.hypergraph, pos={n:n for n in self.hypergraph.nodes}, node_size=10, ax=ax, cmap='Set3')
            