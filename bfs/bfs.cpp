#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set** frontierset,
    int* distances)
{
    int chunkset = 1000;

    if(g->num_nodes <= 1000)
        chunkset = 500;


    #pragma omp parallel for schedule(dynamic, chunkset)
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER) {

                if(__sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, distances[node] + 1)) {

                    int index;
                    int threadid = omp_get_thread_num();
                    index = frontierset[threadid]->count++;

                    frontierset[threadid]->vertices[index] = outgoing;
                }
            }
        }
    }
}

void bottom_up_step(
    Graph g,
    vertex_set** frontierset,
    int* flags,
    int* newflags,
    int* distances)
{
    //iterate through all nodes
    int chunkset = 100000;

    if(g->num_nodes <= 1000)
        chunkset = 100;
    

    #pragma omp parallel for schedule(dynamic, chunkset)
    for(int i = 0; i < g->num_nodes; i++) {
        if(distances[i] == NOT_VISITED_MARKER) {
            //get incoming edges to node
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[i + 1];

            //iterate through all neighbors
            for(int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                //if neighbor is in the frontier and not visited
                //update distance and count on new frontier
                //add to new frontier and update newflags

                if(flags[incoming] == 1) {
                    distances[i] = distances[incoming] + 1;
                    //if(__sync_bool_compare_and_swap(distances + i, NOT_VISITED_MARKER, distances[incoming] + 1)) {}

                    int threadid = omp_get_thread_num();
                    int index = frontierset[threadid]->count++;

                    frontierset[threadid]->vertices[index] = i;

                    newflags[i] = 1;
                    //above causes segfault
                    
                }
            }
        }
    }
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    int maxthread = omp_get_max_threads();

    vertex_set* frontier = &list1;
    vertex_set** frontierset = new vertex_set*[maxthread];
    vertex_set* lists = new vertex_set[maxthread];

    for(int i = 0; i < maxthread; i++) {
        vertex_set_init(lists + i, graph->num_nodes);
        frontierset[i] = lists + i;
    }


    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        for(int i = 0; i < maxthread; i++) {
            vertex_set_clear(frontierset[i]);
        }

        top_down_step(graph, frontier, frontierset, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        for(int i = 1; i < maxthread; i++) {
            memcpy(frontierset[0]->vertices + frontierset[0]->count, frontierset[i]->vertices, frontierset[i]->count * sizeof(int));
            frontierset[0]->count += frontierset[i]->count;
        }
        vertex_set* tmp = frontier;
        frontier = frontierset[0];
        frontierset[0] = tmp;
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    int maxthread = omp_get_max_threads();

    vertex_set* frontier = &list1;
    vertex_set** frontierset = new vertex_set*[maxthread];
    vertex_set* lists = new vertex_set[maxthread];

    for(int i = 0; i < maxthread; i++) {
        vertex_set_init(lists + i, graph->num_nodes);
        frontierset[i] = lists + i;
    }


    int* flags = (int*) calloc(graph->num_nodes, sizeof(int));
    int* newflags = (int*) calloc(graph->num_nodes, sizeof(int));

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    flags[ROOT_NODE_ID] = 1;


    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        for(int i = 0; i < maxthread; i++) {
            vertex_set_clear(frontierset[i]);
        }


        bottom_up_step(graph, frontierset, flags, newflags, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        for(int i = 1; i < maxthread; i++) {
            memcpy(frontierset[0]->vertices + frontierset[0]->count, frontierset[i]->vertices, frontierset[i]->count * sizeof(int));
            frontierset[0]->count += frontierset[i]->count;
        }
        vertex_set* tmp = frontier;
        frontier = frontierset[0];
        frontierset[0] = tmp;

        int* temp = flags;
        flags = newflags;
        newflags = temp;

        
        #pragma omp parallel for
        for(int i = 0; i < graph->num_nodes; i++) {
            newflags[i] = 0;
        }
    }


}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    int maxthread = omp_get_max_threads();

    vertex_set* frontier = &list1;
    vertex_set** frontierset = new vertex_set*[maxthread];
    vertex_set* lists = new vertex_set[maxthread];

    #pragma omp parallel for
    for(int i = 0; i < maxthread; i++) {
        vertex_set_init(lists + i, graph->num_nodes);
        frontierset[i] = lists + i;
    }

    int alpha = 14, beta = 24;

    int* flags = (int*) calloc(graph->num_nodes, sizeof(int));
    int* newflags = (int*) calloc(graph->num_nodes, sizeof(int));

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
        flags[i] = 0;
        newflags[i] = 0;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    //start with top down step
    top_down_step(graph, frontier, frontierset, sol->distances);
    bool TB = true;
    for(int i = 1; i < maxthread; i++) {
        memcpy(frontierset[0]->vertices + frontierset[0]->count, frontierset[i]->vertices, frontierset[i]->count * sizeof(int));
        frontierset[0]->count += frontierset[i]->count;
    }
    vertex_set* tmp = frontier;
    frontier = frontierset[0];
    frontierset[0] = tmp;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        for(int i = 0; i < maxthread; i++) {
            vertex_set_clear(frontierset[i]);
        }


        if(TB) {
            int mf = 0;
            int mu = 0;
            #pragma omp parallel for reduction(+:mf, mu)
            for(int i = 0; i < graph->num_nodes; i++) {
                if(flags[i] == 1) {
                    mf += outgoing_size(graph, i);
                }
                if(sol->distances[i] != NOT_VISITED_MARKER) {
                    mu += incoming_size(graph, i);
                }
            }
            if(mf > mu/alpha) {
                TB = false;
                #pragma omp parallel for
                for(int j = 0; j < frontier->count; j++) {
                    int vert = frontier->vertices[j];
                    flags[vert] = 1;
                }
                bottom_up_step(graph, frontierset, flags, newflags, sol->distances);
            } else {
                top_down_step(graph, frontier, frontierset, sol->distances);
            }
        } else {
            if(frontier->count < (graph->num_nodes)/beta) {
                TB = true;
                top_down_step(graph, frontier, frontierset, sol->distances);
            } else {
                bottom_up_step(graph, frontierset, flags, newflags, sol->distances);
            }
            
        }

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        for(int i = 1; i < maxthread; i++) {
            memcpy(frontierset[0]->vertices + frontierset[0]->count, frontierset[i]->vertices, frontierset[i]->count * sizeof(int));
            frontierset[0]->count += frontierset[i]->count;
        }
        vertex_set* tmp = frontier;
        frontier = frontierset[0];
        frontierset[0] = tmp;

        int* temp = flags;
        flags = newflags;
        newflags = temp;

        
        #pragma omp parallel for
        for(int i = 0; i < graph->num_nodes; i++) {
            newflags[i] = 0;
        }
    }


}
