---
layout: post
title: Passwordless login linux system
author: Saeid Amiri
date: 2021-11-2
description: Passwordless login linux system
published: true
tags: ssh
categories: DataScience
comments: false
---

# Introduction
We often use ssh (Secure SHELL) to connect server via terminal remotely. If you are tired  entering password to login the server, just do the following steps to enter passwordless.

## Add server info to config file
Open the ssh config file on your PC (ssh client)
```
vim \$HOME/.ssh/config
```

Add a name, the server ip (hostname), and the user id on server to the config file:
```
Host server1
HostName 8.8.8.8
User usr1
```

##  generate ssh key
Run the following code to create new keys. 
```
ssh-keygen -t rsa
```

## copy key to server
Run the follwoing code to copy key to server1, and add enter password
```
ssh-copy-id server1
```

Now you should be able to enter the server passwordlessly. 
```
ssh server1
```


### License
Copyright (c) 2021 Saeid Amiri