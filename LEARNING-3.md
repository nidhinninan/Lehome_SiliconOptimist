# assigned user permission correctly

- probably don't need below cmd any more as I have updated the .sh file (had the use principia instead of $USER in the .sh script to correctly give permissions)
```python
sudo chown -R principia:principia /data/lehome_workspace
```
- use rsync with the '--exclude tags' because we don't want to loss the auto generated files because of uv sync and other cmds from the install .sh file.
 
