#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LENGTH 50
#define MAX_STATIONS 100
#define MAX_LINES 50

// Represents a connection between two stations and its fare
typedef struct {
  char source[MAX_LENGTH];
  char destination[MAX_LENGTH];
  int fare;
} Edge;

// Represents a metro line with its name and sequence of stations
typedef struct {
  char line_name[MAX_LENGTH];
  int station_count;
  char stations[MAX_STATIONS][MAX_LENGTH];
} Line;

// Helper struct to track stations that intersect multiple lines
typedef struct {
  char station[MAX_LENGTH];
  int line_count;
} StationIntersection;

// Find index of a station in the intersections array
// Returns -1 if not found
int find_station_index(StationIntersection *stations, int count,
                       const char *station_name) {
  for (int i = 0; i < count; i++) {
    if (strcmp(stations[i].station, station_name) == 0) {
      return i;
    }
  }
  return -1;
}

// Shows which lines a particular station belongs to
void display_station_details(const char *station_name, Line *lines,
                             int line_count, Edge *edges, int edge_count) {
  int found_station = 0;

  // Search through all lines to find which ones contain this station
  for (int i = 0; i < line_count; i++) {
    for (int j = 0; j < lines[i].station_count; j++) {
      if (strcmp(lines[i].stations[j], station_name) == 0) {
        if (!found_station) {
          printf("Station: %s\n", station_name);
          found_station = 1;
          printf("Belongs to lines: ");
        }
        printf("%s ", lines[i].line_name);
        printf("\nadjacent stations: \n");
        for (int i = 0; i < edge_count; i++) {
          if (strcmp(lines[i].stations[j], edges[i].source) == 0)
            printf("%s %d\n", edges[i].destination, edges[i].fare);
          else if (strcmp(lines[i].stations[j], edges[i].destination) == 0) {
            printf("%s %d\n", edges[i].source, edges[i].fare);
          }
        }
        break;
      }
    }
  }

  if (!found_station) {
    printf("Station %s not found.\n", station_name);
  } else {
    printf("\n");
  }
}

// Finds and displays all stations where multiple lines intersect
void display_intersections(Line *lines, int line_count) {
  StationIntersection intersections[MAX_STATIONS];
  int intersection_count = 0;

  for (int i = 0; i < line_count; i++) {
    for (int j = 0; j < lines[i].station_count; j++) {
      int index = find_station_index(intersections, intersection_count,
                                     lines[i].stations[j]);
      if (index == -1) {
        // New station found
        strcpy(intersections[intersection_count].station, lines[i].stations[j]);
        intersections[intersection_count].line_count = 1;
        intersection_count++;
      } else {
        // Existing station found on another line
        intersections[index].line_count++;
      }
    }
  }

  // Filter out stations that only appear on one line
  StationIntersection multi_line_stations[MAX_STATIONS];
  int multi_line_count = 0;
  for (int i = 0; i < intersection_count; i++) {
    if (intersections[i].line_count > 1) {
      multi_line_stations[multi_line_count++] = intersections[i];
    }
  }

  // Sort intersections by number of lines (descending)
  for (int i = 0; i < multi_line_count - 1; i++) {
    for (int j = i + 1; j < multi_line_count; j++) {
      if (multi_line_stations[i].line_count <
          multi_line_stations[j].line_count) {
        StationIntersection temp = multi_line_stations[i];
        multi_line_stations[i] = multi_line_stations[j];
        multi_line_stations[j] = temp;
      }
    }
  }

  // Display results
  printf("Intersections:\n");
  for (int i = 0; i < multi_line_count; i++) {
    printf("%s, %d\n", multi_line_stations[i].station,
           multi_line_stations[i].line_count);
  }
}

// Represents a node in our BFS search
typedef struct {
  char station[MAX_LENGTH];
  int total_cost;
  char path[MAX_STATIONS][MAX_LENGTH];
  int path_len;
} BFSNode;

// Uses BFS to find the nearest intersection from a given station
void find_path_to_nearest_intersection(const char *start_station, Edge *edges,
                                       int edge_count, Line *lines,
                                       int line_count) {
  // First, verify the station exists in our network
  int found_start = 0;
  for (int i = 0; i < edge_count; i++) {
    if (strcmp(edges[i].source, start_station) == 0 ||
        strcmp(edges[i].destination, start_station) == 0) {
      found_start = 1;
      break;
    }
  }

  if (!found_start) {
    printf("Station %s not found in the network.\n", start_station);
    return;
  }

  // Initialize BFS queue
  BFSNode queue[MAX_STATIONS];
  int front = 0, rear = 0;

  // Set up starting node
  strcpy(queue[rear].station, start_station);
  queue[rear].total_cost = 0;
  queue[rear].path_len = 1;
  strcpy(queue[rear].path[0], start_station);
  rear++;

  int found_intersection = 0;

  // Begin BFS
  while (front < rear && !found_intersection) {
    BFSNode current = queue[front++];

    // Check all possible connections from current station
    for (int i = 0; i < edge_count; i++) {
      if (strcmp(edges[i].source, current.station) == 0 ||
          strcmp(edges[i].destination, current.station) == 0) {
        const char *next_station = strcmp(edges[i].source, current.station) == 0
                                       ? edges[i].destination
                                       : edges[i].source;

        // Check if we've already visited this station
        int is_visited = 0;
        for (int j = 0; j < current.path_len; j++) {
          if (strcmp(current.path[j], next_station) == 0) {
            is_visited = 1;
            break;
          }
        }
        if (is_visited)
          continue;

        // Create new path including this station
        BFSNode new_node = current;
        strcpy(new_node.station, next_station);
        new_node.total_cost += edges[i].fare;
        strcpy(new_node.path[new_node.path_len++], next_station);

        // Check if this station is an intersection
        int line_count_at_station = 0;
        for (int j = 0; j < line_count; j++) {
          for (int k = 0; k < lines[j].station_count; k++) {
            if (strcmp(lines[j].stations[k], next_station) == 0) {
              line_count_at_station++;
              break;
            }
          }
          if (line_count_at_station > 1)
            break;
        }

        if (line_count_at_station > 1) {
          // We found an intersection! Print the path
          printf("Path to nearest intersection: ");
          for (int j = 0; j < new_node.path_len; j++) {
            printf("%s", new_node.path[j]);
            if (j < new_node.path_len - 1) {
              printf(" -> ");
            }
          }
          printf(", Cost: %d\n", new_node.total_cost);
          found_intersection = 1;
          break;
        }

        queue[rear++] = new_node;
      }
    }
  }

  if (!found_intersection) {
    printf("No intersection found from station %s.\n", start_station);
  }
}

// Helper function for DFS route finding
void dfs_find_routes(const char *current_station, const char *destination,
                     int *visited, char path[MAX_STATIONS][MAX_LENGTH],
                     int *path_len, int total_cost, Edge *edges,
                     int edge_count) {
  // Add current station to path
  strcpy(path[*path_len], current_station);
  (*path_len)++;

  // Check if we've reached our destination
  if (strcmp(current_station, destination) == 0) {
    printf("Path: ");
    for (int i = 0; i < *path_len; i++) {
      printf("%s", path[i]);
      if (i < *path_len - 1) {
        printf(" -> ");
      }
    }
    printf(", Cost: %d\n", total_cost);
  } else {
    // Try all possible next stations
    for (int i = 0; i < edge_count; i++) {
      const char *next_station = NULL;
      int edge_cost = 0;

      if (strcmp(edges[i].source, current_station) == 0) {
        next_station = edges[i].destination;
        edge_cost = edges[i].fare;
      } else if (strcmp(edges[i].destination, current_station) == 0) {
        next_station = edges[i].source;
        edge_cost = edges[i].fare;
      }

      if (next_station != NULL && !visited[i]) {
        visited[i] = 1;
        dfs_find_routes(next_station, destination, visited, path, path_len,
                        total_cost + edge_cost, edges, edge_count);
        visited[i] = 0;
      }
    }
  }

  (*path_len)--;
}

// Find all possible routes between two stations using DFS
void find_routes(const char *source, const char *destination, Edge *edges,
                 int edge_count) {
  int visited[MAX_STATIONS] = {0};
  char path[MAX_STATIONS][MAX_LENGTH];
  int path_len = 0;

  dfs_find_routes(source, destination, visited, path, &path_len, 0, edges,
                  edge_count);
}

// Display the first and last stations of each line
void display_terminal_stations(Line *lines, int line_count) {
  for (int i = 0; i < line_count; i++) {
    if (lines[i].station_count > 0) {
      printf("%s : %s, %s\n", lines[i].line_name, lines[i].stations[0],
             lines[i].stations[lines[i].station_count - 1]);
    }
  }
}

int main() {
  Edge *edges = NULL;
  Line *lines = NULL;
  int edge_count = 0, line_count = 0;

  int choice;
  do {
    printf("\n--- Metro Network Menu ---\n");
    printf("1. Enter Edges\n");
    printf("2. Enter Metro Lines\n");
    printf("3. Display Metro Network Edges\n");
    printf("4. Display Station details\n");
    printf("5. Display Stations with Multiple Intersections\n");
    printf("6. Find Path to Nearest Intersection\n");
    printf("7. Find Possible Routes\n");
    printf("8. Display Terminal Stations\n");
    printf("9. Exit\n");
    printf("Enter your choice: ");
    scanf("%d", &choice);

    switch (choice) {
    case 1: {
      printf("Enter the number of edges: ");
      scanf("%d", &edge_count);

      edges = (Edge *)malloc(edge_count * sizeof(Edge));
      if (!edges) {
        printf("Memory allocation failed for edges.\n");
        return 1;
      }

      printf("Enter the edges (source destination fare):\n");
      for (int i = 0; i < edge_count; i++) {
        scanf("%s %s %d", edges[i].source, edges[i].destination,
              &edges[i].fare);
      }
      break;
    }

    case 2: {
      printf("Enter the number of metro lines: ");
      scanf("%d", &line_count);

      lines = (Line *)malloc(line_count * sizeof(Line));
      if (!lines) {
        printf("Memory allocation failed for lines.\n");
        free(edges);
        return 1;
      }

      for (int i = 0; i < line_count; i++) {
        printf("Enter line name: ");
        scanf("%s", lines[i].line_name);

        printf("Enter the number of stations in line %s: ", lines[i].line_name);
        scanf("%d", &lines[i].station_count);

        if (lines[i].station_count == 0) {
          printf("No stations found for line %s.\n", lines[i].line_name);
          continue;
        }

        printf("Enter the stations:\n");
        for (int j = 0; j < lines[i].station_count; j++) {
          scanf("%s", lines[i].stations[j]);
        }
      }
      break;
    }

    case 3: {
      printf("\nMetro Network Edges:\n");
      for (int i = 0; i < edge_count; i++) {
        printf(" %s -> %s, Fare: %d\n", edges[i].source, edges[i].destination,
               edges[i].fare);
      }
      break;
    }

    case 4: {
      char station_name[50];
      printf("\nEnter the station name to get its details: ");
      scanf("%s", station_name);
      printf("\nStation Details:\n");
      display_station_details(station_name, lines, line_count, edges,
                              edge_count);
      break;
    }

    case 5: {
      display_intersections(lines, line_count);
      break;
    }

    case 6: {
      char station[MAX_LENGTH];
      printf("Enter station name: ");
      scanf("%s", station);
      find_path_to_nearest_intersection(station, edges, edge_count, lines,
                                        line_count);
      break;
    }

    case 7: {
      char source[MAX_LENGTH], destination[MAX_LENGTH];
      printf("Enter source station: ");
      scanf("%s", source);
      printf("Enter destination station: ");
      scanf("%s", destination);
      find_routes(source, destination, edges, edge_count);
      break;
    }

    case 8: {
      display_terminal_stations(lines, line_count);
      break;
    }

    case 9: {
      printf("Exiting the program.\n");
      break;
    }

    default:
      printf("Invalid choice. Please try again.\n");
    }
  } while (choice != 9);

  // Clean up
  free(edges);
  free(lines);
  return 0;
}
