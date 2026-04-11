# Classifier
- For lazy loading the classifier, we only need to load it once every episode so the policy switching is not too big a concern as far as time loss goes.
- Other approach; load all 4 on the VRAM so that can don't have the load from "memory to VRAM" overhead

# KArpathy's autoResearch
- time contsrainted wall clock anaylsis