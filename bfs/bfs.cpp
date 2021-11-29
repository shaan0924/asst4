#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <iostream>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
//#define VERBOSE 1

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
    int* flags,
    int* newflags,
    int& count,
    int* distances)
{
    int chunkset = 100000;

    if(g->num_nodes <= 1000000)
        chunkset = 10000;


    #pragma omp parallel for schedule(dynamic, chunkset) reduction(+:count)
    for (int i=0; i<g->num_nodes; i++) {
        if(flags[i] == 1) {

            int start_edge = g->outgoing_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[i + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER) {

                    if(__sync_bool_compare_and_swap(distances + outgoing, NOT_VISITED_MARKER, distances[i] + 1)) {

                        newflags[outgoing] = 1;
                        count++;
                    }
                }
            }
        }
    }
}

void bottom_up_step(
    Graph g,
    int* flags,
    int* newflags,
    int& count,
    int* distances)
{
    //iterate through all nodes
    int chunkset = 100000;

    if(g->num_nodes <= 1000000)
        chunkset = 10000;
    

    #pragma omp parallel for schedule(dynamic, chunkset) reduction(+:count)
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
                    
                    newflags[i] = 1;

                    count++;
                    break;
                    
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

    int* flags = (int*) calloc(graph->num_nodes, sizeof(int));
    int* newflags = (int*) calloc(graph->num_nodes, sizeof(int));


    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;
    flags[ROOT_NODE_ID] = 1;

    int count = 1;

    while (count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        count = 0;

        top_down_step(graph, flags, newflags, count, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", count, end_time - start_time);
#endif

        // swap pointers

        int* temp = flags;
        flags = newflags;
        newflags = temp;

        
        #pragma omp parallel for
        for(int i = 0; i < graph->num_nodes; i++) {
            newflags[i] = 0;
        }


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


    int* flags = (int*) calloc(graph->num_nodes, sizeof(int));
    int* newflags = (int*) calloc(graph->num_nodes, sizeof(int));

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;
    flags[ROOT_NODE_ID] = 1;

    int count = 1;


    while (count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        count = 0;

        bottom_up_step(graph, flags, newflags, count, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", count, end_time - start_time);
#endif

        

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

    int alpha = 14;

    int* flags = (int*) calloc(graph->num_nodes, sizeof(int));
    int* newflags = (int*) calloc(graph->num_nodes, sizeof(int));

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i=0; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;
    flags[ROOT_NODE_ID] = 1;

    int count = 0;

    //start with top down step
    top_down_step(graph, flags, newflags, count, sol->distances);
    bool TB = true;
    int* temp = flags;
    flags = newflags;
    newflags = temp;

        
    #pragma omp parallel for
    for(int i = 0; i < graph->num_nodes; i++) {
        newflags[i] = 0;
    }

    while (count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif


        if(count > graph->num_nodes/alpha) {
            TB = false;
            count = 0;
            bottom_up_step(graph, flags, newflags, count, sol->distances);

        } else {
            TB = true;
            count = 0;
            top_down_step(graph, flags, newflags, count, sol->distances);
        }
        

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", count, end_time - start_time);
#endif

        int* tmp = flags;
        flags = newflags;
        newflags = tmp;

        
        #pragma omp parallel for
        for(int i = 0; i < graph->num_nodes; i++) {
            newflags[i] = 0;
        }
        
    }


}
