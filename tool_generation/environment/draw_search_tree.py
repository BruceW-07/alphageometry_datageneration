import networkx as nx
import matplotlib.pyplot as plt

class DrawSearchTree:
    def __init__(self):
        self.graph = nx.DiGraph()
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()  # Set up the plot
        self.pos = {}  # Store the position of nodes

    def add_node(self, name, parent=None, large_textbox=False):
        """Add a node to the graph with an optional edge from a parent."""
        if parent:
            self.graph.add_edge(parent, name)  # Add an edge if there is a parent
        else:
            self.graph.add_node(name)  # Add a node if no parent is specified
        if large_textbox:
            self.pos[name] = None  # Indicate this is a custom large node
        else:
            self.pos[name] = ''
        self.update_plot()  # Update the plot each time a node is added

    def update_plot(self):
        """Clear the current plot, redraw the graph, and add large textboxes if necessary."""
        self.ax.clear()  # Clear the current plot
        pos = nx.spring_layout(self.graph)  # Position nodes using the Spring layout

        # Draw the graph without the large textbox nodes
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000,
                font_size=9, ax=self.ax)

        # Handle large textbox nodes separately
        for node, position in pos.items():
            if self.pos.get(node) is None:  # Check if this is a large textbox node
                # Draw a large square-shaped textbox with a lot of text
                self.draw_large_textbox(position, node)

        plt.pause(0.5)  # Pause to update the plot and give time to view changes
        self.fig.canvas.draw()  # Ensure the canvas is updated

    def draw_large_textbox(self, position, text):
        """Draw a large square-shaped textbox at the specified position."""

        self.ax.text(position[0], position[1], text,
                     fontsize=9, ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=1", facecolor='lightgreen', edgecolor='black'))

    def close(self):
        """Disable interactive mode and show the final plot."""
        plt.ioff()  # Turn off interactive mode

# Usage example
if __name__ == "__main__":
    tree = DrawSearchTree()
    tree.add_node('Root')
    tree.add_node('Child1', 'Root')
    tree.add_node('Large Node with a lot of text', 'Root', large_textbox=True)  # Large textbox node
    print("Press Enter to finish...")
    input()  # Keep the plot window open until the user decides to close it
    tree.close()  # Properly close the plot
