# Cluster Classification

The cluster classification process accepts a list of clusters and data from the
hadron calorimiter and sorts them into hadronic and electromagnetic events.

Each cluster will be processed in parallel.

## Incoming data:

**Phi-Eta Position** of the cluster under analysis. This information is required
to filter the HCAL Tower data, as not all of the towers are needed to determine
the type of the given cluster.

**HCAL tower data.** This is the primary metric that will be used to evaluate
the classification of the cluster. Only the 9 towers surrounding the cluster's
Phi-Eta position are used in evaluation.

This information will be processed in 18 nodes. Nine of the nodes represent the
energies deposited in the towers. The other nine nodes provide deposition
data along the four layers of the HCAL tower.

**Cluster Energy.** This is the total electromagnetic energy from the cluster.

## Outgoing data

The outgoing data provides the relative certainties that the cluster is
electronic or hadronic.
