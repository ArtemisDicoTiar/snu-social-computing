"""
this python file solves exercise from network book - exercise 3.7
"""


"""
2. Consider the graph in Figure 3.21, in which each edge — except the edge connecting
b and c — is labeled as a strong tie (S) or a weak tie (W).
According to the theory of strong and weak ties, with the strong triadic closure assumption, how would you expect the edge connecting b and c to be labeled? Give a
brief (1-3 sentence) explanation for your answer.

The connection between B and C must be weak tie.

Let's assume that the connection between B and C is strong tie.
For node B, it is strongly tied with node E and C this must result at least weak tie between node E and C.
Similarly for node C, as it is strongly tied with node B and F, node B and F must have at least weak tie.
However, the tie between E and C, B and F is not constructed, the assumption violates the triadic closure property.

Therefore, the connection between B and C is weak tie. 
"""

"""
3. In the social network depicted in Figure 3.22, with each edge labeled as either a strong
or weak tie, which nodes satisfy the Strong Triadic Closure Property from Chapter 3,
and which do not? Provide an explanation for your answer.

There are 5 nodes on the Figure 3.22. Let's go around each node to see whether it satisfies the triadic closure property.

i) node A (O)
The node A is having strong tie with node B and D.
There is a weak tie between node B and D.
The triadic closure assumption is satisfied.

ii) node B (O)
There is strong tie with node A and C.
The weak tie is constructed between node A and C.
The triadic closure property is satisfied.

iii) node C
There is strong tie with node B and E.
However, there is no connection between node B and E directly.
This violates the triadic closure assumption.

iv) node D (O)
There is strong tie with node A and E.
There is weak tie between node A and E.
This satisfies triadic closure property.

v) node E
Similar to node C, although there is strong tie with node C and D, no connection is constructed between node C and D.
The triadic closure is violated. 

Therefore, node A, B and D satisfies the property.
"""

"""
4. In the social network depicted in Figure 3.23 with each edge labeled as either a strong
or weak tie, which two nodes violate the Strong Triadic Closure Property? Provide an
explanation for your answer.

The network described on Figure 3.23 is identical to Figure 3.22.
Therefore the node C and E violates the property.

"""

"""
5. In the social network depicted in Figure 3.24, with each edge labeled as either a strong
or weak tie, which nodes satisfy the Strong Triadic Closure Property from Chapter 3,
and which do not? Provide an explanation for your answer.

Similar question is given, similar solution will be given.

i) node A
The strong tie with node B and C.
There is a strong tie between node B and C.
This satisfies the property.

ii) node B
Same triangle as node A.
Satisfies the property.

iii) node C
The node C has strong tie with node A, B, and E.
However, there is no connection between node A and E.
This violates the triadic closure assumption.

iv) node D
The node D only has weak tie.
Nothing to consider further.

v) node E
The node E has only one strong tie.
No more further investigation required.

In a nut shell, node A, B, D and E satisfies the property.

"""




