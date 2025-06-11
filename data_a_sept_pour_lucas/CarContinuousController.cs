using System;
using System.Collections.Generic;
using System.Net.Sockets;
using TMPro;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;
using System.IO;

public class CarContinuousController : Agent
{
    public RawImage carVisionImage;
    public CarController carController;
    public Camera carVisionCamera;
    public Camera carVisionCamera2;
    public GameObject canvas;
    public Transform startPosition;
    public TrackDropDown trackDropDown;
    public BehaviorParameters behaviorParameters;
    public Raycast Raycast;

    [Header("Screen Capture Settings")]
    public bool captureEnabled = true;
    public int captureWidth = 347;
    public int captureHeight = 256;
    public string folderName = "data";
    public string filePrefix = "frame_";
    public string fileExtension = ".png";

    private readonly Dictionary<string, Func<float, string>> _floatActions;
    private readonly List<string> _touchedCheckpoints = new();
    private readonly Dictionary<string, Func<string>> _voidActions;
    private bool _isRunning;
    private RenderTexture _renderTexture;
    private TcpListener _server;
    private TextMeshProUGUI _textMesh;
    private GameObject _textMeshGo;
    private float _timer;
    
    private string _dataPath;
    private int _frameCounter = 0;
    private RenderTexture _captureRenderTexture;
    private Texture2D _screenshot;
    
    public bool resetCarPosition { get; set; }

    public int NumberCollider { get; set; }
    public float Fov { get; set; }

    public int NbRay { get; set; }

    public int CarIndex
    {
        get => behaviorParameters.TeamId;
        set
        {
            behaviorParameters.BehaviorName += value;
            behaviorParameters.TeamId = value;
        }
    }

    private void Start()
    {
        _renderTexture = new RenderTexture(347, 256, 1)
        {
            name = CarIndex.ToString()
        };
        carVisionCamera.targetTexture = _renderTexture;
        carVisionCamera2.targetTexture = _renderTexture;

        Random.InitState(DateTime.Now.Millisecond);

        _textMeshGo = new GameObject();
        _textMeshGo.transform.SetParent(canvas.transform);
        _textMeshGo.transform.localPosition = new Vector3(247, 230 - 30 * CarIndex, 0);
        _textMesh = _textMeshGo.AddComponent<TextMeshProUGUI>();
        _textMesh.enableAutoSizing = true;
        _textMesh.color = Color.black;

        InitializeScreenCapture();
    }

    private void InitializeScreenCapture()
    {
        _dataPath = Path.Combine(Application.dataPath, folderName);
        if (!Directory.Exists(_dataPath))
        {
            Directory.CreateDirectory(_dataPath);
            Debug.Log($"Dossier créé : {_dataPath}");
        }

        _captureRenderTexture = new RenderTexture(captureWidth, captureHeight, 24);
        _screenshot = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);

        Debug.Log($"Capture initialisée pour Agent {CarIndex}. Dossier : {_dataPath}");
    }

    private void Update()
    {
        UpdateTimer();
        
        if (captureEnabled && carVisionCamera != null)
        {
            TakeScreen(carVisionCamera);
            TakeScreen(carVisionCamera2);
        }
    }

    public void TakeScreen(Camera camera)
    {
        RenderTexture previousRenderTexture = camera.targetTexture;
        RenderTexture previousActiveTexture = RenderTexture.active;

        camera.targetTexture = _captureRenderTexture;
        
        camera.Render();
        
        RenderTexture.active = _captureRenderTexture;
        _screenshot.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        _screenshot.Apply();

        camera.targetTexture = previousRenderTexture;
        RenderTexture.active = previousActiveTexture;

        SaveScreenshot();
        
        _frameCounter++;
    }

    private void SaveScreenshot()
    {
        byte[] imageData = _screenshot.EncodeToPNG();
        string fileName = $"{filePrefix}agent{CarIndex}_{_frameCounter:D6}{fileExtension}";
        string filePath = Path.Combine(_dataPath, fileName);
        
        File.WriteAllBytes(filePath, imageData);
        
        if (_frameCounter % 60 == 0)
        {
            Debug.Log($"Agent {CarIndex} - Capture sauvegardée : {fileName} (Frame {_frameCounter})");
        }
    }

    public void ToggleCapture()
    {
        captureEnabled = !captureEnabled;
        Debug.Log($"Agent {CarIndex} - Capture {(captureEnabled ? "activée" : "désactivée")}");
    }

    public void ResetFrameCounter()
    {
        _frameCounter = 0;
        Debug.Log($"Agent {CarIndex} - Compteur de frames remis à zéro");
    }

    private void FixedUpdate()
    {
        if (transform.position.y < -20)
        {
            resetCarPosition = true;
            EndEpisode();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Lines"))
        {
            resetCarPosition = true;
            EndEpisode();
        }
        else if (other.CompareTag("Checkpoint"))
        {
            if (!_touchedCheckpoints.Contains(other.name)) _touchedCheckpoints.Add(other.name);
        }
        else if (other.CompareTag("Finish"))
        {
            if (_touchedCheckpoints.Count == NumberCollider && NumberCollider != 0)
            {
                _timer += Time.deltaTime;
                var minutes = Mathf.FloorToInt(_timer / 60);
                var seconds = Mathf.FloorToInt(_timer % 60);
                Debug.Log($"you finished a lap in {minutes:00}:{seconds:00} !!");
                trackDropDown.UpdateBestScore(_timer);
                EndEpisode();
            }

            _touchedCheckpoints.Clear();
        }
    }

    ~CarContinuousController()
    {
        Destroy(_textMeshGo);
        Destroy(_textMesh);
        
        // Nettoyer les ressources de capture
        if (_captureRenderTexture != null)
        {
            _captureRenderTexture.Release();
        }
        
        if (_screenshot != null)
        {
            DestroyImmediate(_screenshot);
        }
    }

    public void ResetCarPosition()
    {
        carController.Reset();
        gameObject.transform.position = startPosition.position;
        gameObject.transform.rotation = startPosition.rotation;
        transform.Rotate(new Vector3(0, -90, 0));
    }

    private void UpdateTimer()
    {
        _timer += Time.deltaTime;
        _textMesh.text =
            string.Format($"Agent {CarIndex}: {Mathf.FloorToInt(_timer / 60):00}:{Mathf.FloorToInt(_timer % 60):00}");
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var throttle = actionBuffers.ContinuousActions[0];
        var steering = actionBuffers.ContinuousActions[1];
        carController.Move(throttle);
        carController.Turn(steering);
    }

    public override void OnEpisodeBegin()
    {
        _timer = 0f;
        _touchedCheckpoints.Clear();
        if (resetCarPosition)
            ResetCarPosition();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        var (distance, newTexture) = Raycast.GetRaycasts(carVisionCamera.targetTexture,
            carVisionImage.texture as Texture2D, NbRay, Fov);

        carVisionImage.texture = newTexture;

        foreach (var i in distance) sensor.AddObservation(i);

        for (var i = distance.Count; i < behaviorParameters.BrainParameters.VectorObservationSize - 5; i++)
            sensor.AddObservation(-1);
        sensor.AddObservation(carController.Speed());
        sensor.AddObservation(carController.Steering());
        sensor.AddObservation(carController.transform.position.x);
        sensor.AddObservation(carController.transform.position.y);
        sensor.AddObservation(carController.transform.position.z);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // var continuousActions = actionsOut.ContinuousActions;

        // continuousActions[0] = Input.GetAxis("Vertical");
        // continuousActions[1] = Input.GetAxis("Horizontal");
    }

    void OnDestroy()
    {
        // Nettoyer les ressources de capture
        if (_captureRenderTexture != null)
        {
            _captureRenderTexture.Release();
        }
        
        if (_screenshot != null)
        {
            DestroyImmediate(_screenshot);
        }
    }

    void OnApplicationPause(bool pauseStatus)
    {
        // Arrêter la capture quand l'application est en pause
        if (pauseStatus)
        {
            captureEnabled = false;
        }
    }
}
