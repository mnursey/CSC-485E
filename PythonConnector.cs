using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace SchoolProject
{
    public class PythonConnector : MonoBehaviour
    {
        [Range(0, 5000)]
        public float cameraMoveSpeed = 5f;

        public Rigidbody cameraFollowRb;
        Vector3 cameraFollowOriginalPos;

        [HideInInspector] public bool isTxStarted = false;

        public List<PrefabLink> prefabs;
        [Range(1, 50)]
        public float positionScale;

        [Header("Server Info")]

        [SerializeField] string IP = "127.0.0.1"; // local host

        [SerializeField] int rxPort = 8000; // port to receive data from Python on
        [SerializeField] int txPort = 8001; // port to send data to Python on

        UdpClient client;
        IPEndPoint remoteEndPoint;
        Thread receiveThread; // Receiving Thread

        List<DetectedObjects> detectedObjects = new List<DetectedObjects>();

        float xInput;
        float yInput;

        void Awake()
        {
            // Create remote endpoint (to Matlab) 
            remoteEndPoint = new IPEndPoint(IPAddress.Parse(IP), txPort);

            // Create local client
            client = new UdpClient(rxPort);

            // local endpoint define (where messages are received)
            // Create a new thread for reception of incoming messages
            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();

            // Initialize (seen in comments window)
            print("UDP Comms Initialised");

            cameraFollowOriginalPos = cameraFollowRb.transform.position;
        }

        // Receive data, update packets received
        private void ReceiveData()
        {
            while (true)
            {
                try
                {
                    IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                    byte[] data = client.Receive(ref anyIP);
                    string text = Encoding.UTF8.GetString(data);
                    ProcessInput(text);
                }
                catch (Exception err)
                {
                    print(err.ToString());
                }
            }
        }

        private void ProcessInput(string input)
        {
            // PROCESS INPUT RECEIVED STRING HERE

            if (!isTxStarted) // First data arrived so tx started
            {
                isTxStarted = true;
            }

            detectedObjects.Add(JsonUtility.FromJson<DetectedObjects>(input));
        }

        private void FixedUpdate()
        {
            if(detectedObjects.Count > 0)
            {
                // Remove old objects
                foreach (Transform child in transform)
                {
                    Destroy(child.gameObject);
                }

                // Add new objects
                foreach (DetectedObject detectObject in detectedObjects[0].data)
                {
                    PrefabLink prefabLink = prefabs.Find((x) => x.type == detectObject.type);
                       
                    // Only spawn the object if the type matches a prefab we have
                    if(prefabLink != null)
                    {
                        Instantiate(prefabLink.prefab, new Vector3(-detectObject.x * positionScale, 0f, detectObject.y * positionScale), prefabLink.prefab.transform.rotation, transform);
                    }
                }

                // Reset camera position
                if(detectedObjects[0].userInputSwitch == 0)
                {
                    cameraFollowRb.transform.position = cameraFollowOriginalPos;
                    cameraFollowRb.velocity = Vector3.zero;
                    cameraFollowRb.angularVelocity = Vector3.zero;
                }

                // Normalize input between -1 and 1
                xInput = (detectedObjects[0].userInputX - 512f) / 512f;
                yInput = (detectedObjects[0].userInputY - 512f) / 512f;

                // Remove low values so we don't get a drift
                xInput = Mathf.Abs(xInput) < 0.1 ? 0f : xInput;
                yInput = Mathf.Abs(yInput) < 0.1 ? 0f : yInput;

                detectedObjects.RemoveAt(0);
            }

            // Update camera position
            cameraFollowRb.AddRelativeForce(new Vector3(-xInput, 0f, yInput) * cameraMoveSpeed * Time.fixedDeltaTime);
        }

        //Prevent crashes - close clients and threads properly!
        void OnDisable()
        {
            if (receiveThread != null)
                receiveThread.Abort();

            client.Close();
        }
    }

    [Serializable]
    public class DetectedObjects
    {
        public DetectedObject[] data;
        public int userInputX;
        public int userInputY;
        public int userInputSwitch;
    }

    [Serializable]
    public class DetectedObject
    {
        public float x;
        public float y;
        public float area;
        public string type;

        public Vector2[] points;

        public override string ToString()
        {
            return string.Format("x: {0}, y: {1}, area {2}, type: {3}", x, y, area, type);
        }
    }

    [Serializable]
    public class PrefabLink
    {
        public GameObject prefab;
        public string type;
    }
}

