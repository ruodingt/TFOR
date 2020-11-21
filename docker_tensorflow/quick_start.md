## Quick Start

### install docker compose
[guide](https://docs.docker.com/compose/install/)

### build and start
```
# build container:
USER_ID=$UID docker-compose build detectron2

# start container stack
USER_ID=$UID docker-compose up -d
```

## Feature

user can ssh into the container with 

`ssh appuser@localhost -p 6022` with password `makefog`

Once you successfully get in you will be able to config your ssh interpreter easily

## change dir ownership 
Change ownership if necessary
```bash
sudo chown appuser:sudo detectron2
ls -l
```

[discussion for mount folder with non-root user](https://github.com/moby/moby/issues/2259)


# SCP
scp -P 6022 -r appuser@13.210.155.248:/home/appuser/detectron2/data /home/appuser/detectron2/data  

scp -P 6022 -r appuser@13.210.155.248:/home/appuser/detectron2 /home/appuser/detectron2

scp -P 6022 -r appuser@52.62.91.43:/home/appuser/detectron2 /home/appuser

## Install AWS CLI


