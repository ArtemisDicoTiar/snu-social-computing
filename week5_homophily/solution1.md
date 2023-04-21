2. Given a bipartite affiliation graph, showing the membership of people in different social foci, researchers sometimes create a projected graph on just the people, in which we join two people when they have a focus in common.

(a) Draw what such a projected graph would look like for the example of memberships
on corporate boards of directors from Figure 4.4. Here the nodes would be the seven people in the figure, and there would be an edge joining any two who serve on a board of directors together.
* `images/Q2.png`

(b) Give an example of two different affiliation networks — on the same set of people, but with different foci — so that the projected graphs from these two different affiliation networks are the same. This shows how information can be “lost” when moving from the full affiliation network to just the projected graph on the set of people.
Let's assume there is a new affiliation between 'Al Gore', 'Steve Jobs' and 'Andrea Jung' which is not having focus on either 'Apple' or 'Disney'. Let's assume the new focus is 'OpenAI'.
This will not disrupt the affiliation networks as they already had relationship because of 'Apple'.
However, when reconstructing the affiliation network with new focus 'OpenAI' will be confusing as they had huge relationship of 4 people. This will lose information that the three people are having same focus 'OpenAI'. 

3. Consider the affiliation network in Figure 4.21, with six people labeled A–F, and three foci labeled X, Y , and Z.
(a) Draw the derived network on just the six people as in Exercise 2, joining two people when they share a focus.
* `images/Q3.png`
(b) In the resulting network on people, can you identify a sense in which the triangle on the nodes A, C, and E has a qualitatively different meaning than the other triangles that appear in the network? Explain.
The node A is having focus on X, Y.
The node C is focusing on X, Z.
The node E's focus is Y and Z.
As the node A, C and E are having triadic closure, they all may have same foci, X, Y and Z, in some time after.  

4. Given a network showing pairs of people who share activities, we can try to reconstruct an affiliation network consistent with this data.
For example, suppose that you are trying to infer the structure of a bipartite affiliation network, and by indirect observation you’ve obtained the projected network on just the set of people, constructed as in Exercise 2: there is an edge joining each pair of people who share a focus. This projected network is shown in Figure 4.22.
(a) Draw an affiliation network involving these six people, together with four foci that you should define, whose projected network is the graph shown in Figure 4.22.
* `images/Q4.png`
(b) Explain why any affiliation network capable of producing the projected network in Figure 4.22 must have at least four foci.
To build the affiliation network with minimum foci, node should be considered in descending order of number of connection between other nodes.
Therefore, there can be a foci "k" and "n" which is constructed based on node B and C in projected network.
After put a foci "k" and "n", only two edges are left: A-E, F-D.
In conclusion, minimum 4 foci are needed to construct affiliation network from given projected network.
