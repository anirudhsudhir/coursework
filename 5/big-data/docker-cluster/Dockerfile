FROM ubuntu

WORKDIR /opt

RUN apt update -y
RUN apt install curl openjdk-11-jdk -y

RUN curl -LO https://dlcdn.apache.org/hadoop/common/hadoop-3.4.1/hadoop-3.4.1.tar.gz
RUN tar -xzf hadoop-3.4.1.tar.gz

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64
ENV HADOOP_HOME=/opt/hadoop-3.4.1

EXPOSE 9870 8088 9864 8042 19888

RUN cat > $HADOOP_HOME/etc/hadoop/core-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://namenode:9000</value>
    </property>
</configuration>
EOF

RUN cat > $HADOOP_CONF_DIR/hdfs-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>2</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///opt/hadoop/data/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:///opt/hadoop/data/datanode</value>
    </property>
    <property>
        <name>dfs.namenode.http-address</name>
        <value>namenode:9870</value>
    </property>
    <property>
        <name>dfs.namenode.rpc-address</name>
        <value>namenode:9000</value>
    </property>
    <property>
        <name>dfs.datanode.http.address</name>
        <value>0.0.0.0:9864</value>
    </property>
</configuration>
EOF


RUN echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh

RUN cat > custom-script-start-namenode.sh << 'EOF'
#!/bin/bash
$HADOOP_HOME/bin/hdfs namenode -format
$HADOOP_HOME/bin/hdfs --daemon start namenode
EOF

RUN cat > custom-script-start-datanode.sh << 'EOF'
#!/bin/bash
$HADOOP_HOME/bin/hdfs --daemon start datanode
EOF

RUN cat > custom-script-start-resourcemanager.sh << 'EOF'
#!/bin/bash
$HADOOP_HOME/bin/yarn --daemon start resourcemanager
EOF

RUN cat > custom-script-start-nodemanager.sh << 'EOF'
#!/bin/bash
$HADOOP_HOME/bin/yarn --daemon start nodemanager
EOF

RUN cat > custom-script-start-historyserver.sh << 'EOF'
#!/bin/bash
$HADOOP_HOME/bin/mapred --daemon start historyserver
EOF

CMD ["bash"]
