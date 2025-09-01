import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import styled, { keyframes } from 'styled-components';
import { FiLogOut, FiMenu, FiX } from 'react-icons/fi';
import '../styles/SecurityDashboard.css';
import sciFiLogo from '../assets/sci-fi-logo.png'; // Import as in Login.jsx

// Fallback logo in case sciFiLogo fails to load
const fallbackLogo = 'https://via.placeholder.com/120?text=SciFiLogo';

// Floating animation for logo
const float = keyframes`
  0% { transform: translateY(0); }
  50% { transform: translateY(-20px); }
  100% { transform: translateY(0); }
`;

// Page Layout
const PageWrapper = styled.div`
  min-height: 100vh;
  display: flex;
  background: linear-gradient(135deg, #0d0d1a, #1a1a33);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  position: relative;
`;

// Sidebar
const Sidebar = styled.aside`
  width: 280px;
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  padding: 2rem;
  display: flex;
  flex-direction: column;
  color: #e0e0e0;
  position: fixed;
  height: 100%;
  top: 0;
  left: 0;
  z-index: 1000;
  transform: ${({ isOpen }) => (isOpen ? 'translateX(0)' : 'translateX(-100%)')};
  transition: transform 0.3s ease-in-out;

  @media (max-width: 768px) {
    width: 250px;
  }

  @media (min-width: 769px) {
    transform: translateX(0);
    position: static;
  }
`;

const Logo = styled.img`
  width: 120px;
  height: 120px;
  margin: 0 auto 2rem;
  animation: ${float} 6s ease-in-out infinite;
  display: block; /* Ensure the logo is visible */

  @media (max-width: 768px) {
    width: 100px;
    height: 100px;
  }
`;

const NavButton = styled.button`
  background: none;
  border: none;
  color: #e0e0e0;
  text-align: left;
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: background 0.3s;

  &:hover {
    background: rgba(255, 255, 255, 0.1);
  }

  @media (max-width: 768px) {
    font-size: 1.2rem;
    padding: 1rem;
  }
`;

// Toggle Button for Mobile
const ToggleButton = styled.button`
  position: fixed;
  top: 1rem;
  left: 1rem;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 8px;
  padding: 0.5rem;
  color: #e0e0e0;
  cursor: pointer;
  z-index: 1100;
  display: none;

  @media (max-width: 768px) {
    display: block;
  }
`;

// Main Content
const Main = styled.main`
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
  margin-left: 280px;
  min-height: 100vh;

  @media (max-width: 768px) {
    margin-left: 0;
    padding: 4rem 1rem 2rem;
  }
`;

const Section = styled.section`
  margin-bottom: 2rem;
  background: rgba(255, 255, 255, 0.05);
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  color: #e0e0e0;

  @media (max-width: 768px) {
    padding: 1rem;
  }
`;

const SectionTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #00d9a6;

  @media (max-width: 768px) {
    font-size: 1.25rem;
  }
`;

const Form = styled.form`
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
  margin-bottom: 1rem;

  @media (min-width: 769px) {
    grid-template-columns: 1fr auto;
  }
`;

const Input = styled.input`
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 1rem;
  outline: none;
  width: 100%;
  box-sizing: border-box;

  &::placeholder {
    color: rgba(255, 255, 255, 0.7);
  }

  @media (max-width: 768px) {
    padding: 1rem;
    font-size: 1.1rem;
  }
`;

const FileInput = styled.input`
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 1rem;
  outline: none;
  width: 100%;
  box-sizing: border-box;

  @media (max-width: 768px) {
    padding: 1rem;
    font-size: 1.1rem;
  }
`;

const Button = styled.button`
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #00d9a6, #00c096);
  color: #0d0d1a;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.3s;
  font-size: 1rem;

  &:hover {
    background: linear-gradient(135deg, #00c096, #00d9a6);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  @media (max-width: 768px) {
    padding: 1rem;
    font-size: 1.1rem;
  }
`;

const Table = styled.table`
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  color: #e0e0e0;
`;

const Th = styled.th`
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.1);
  font-weight: 600;
  text-align: left;

  @media (max-width: 768px) {
    padding: 0.5rem;
    font-size: 0.9rem;
  }
`;

const Td = styled.td`
  padding: 0.75rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);

  @media (max-width: 768px) {
    padding: 0.5rem;
    font-size: 0.9rem;
  }
`;

const ErrorText = styled.p`
  color: #ff6b6b;
  margin-bottom: 1rem;

  @media (max-width: 768px) {
    font-size: 0.9rem;
  }
`;

const ResultText = styled.p`
  margin: 0.5rem 0;
  color: #e0e0e0;

  @media (max-width: 768px) {
    font-size: 0.9rem;
  }
`;

const Highlight = styled.span`
  color: #00d9a6;
`;

const StatusText = styled.span`
  color: ${({ authorized }) => (authorized ? '#00d9a6' : '#ff6b6b')};
`;

const SpinnerOverlay = styled.div`
  display: flex;
  justify-content: center;
  margin-top: 1rem;
`;

const Spinner = styled.div`
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-top: 4px solid #00d9a6;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const SecurityDashboard = () => {
  const navigate = useNavigate();
  const [latestLog, setLatestLog] = useState(null);
  const [plateNumber, setPlateNumber] = useState('');
  const [verifyResult, setVerifyResult] = useState(null);
  const [error, setError] = useState('');
  const [captureResult, setCaptureResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const API_BASE_URL = 'http://192.168.0.150:8000';

  useEffect(() => {
    const token = localStorage.getItem('token');
    console.log(`SecurityDashboard - Token found: ${token}`);
    if (!token) {
      console.log('No token found, redirecting to login...');
      navigate('/login');
      return;
    }

    console.log(`Fetching latest log from ${API_BASE_URL}/logs?limit=1`);
    fetchLatestLog();
    const interval = setInterval(fetchLatestLog, 5000);
    return () => clearInterval(interval);
  }, [navigate]);

  const fetchLatestLog = async () => {
    try {
      const token = localStorage.getItem('token');
      console.log(`Sending request to ${API_BASE_URL}/logs?limit=1 with token: ${token}`);
      const response = await axios.get(`${API_BASE_URL}/logs?limit=1`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      console.log('Latest log response:', response.data);
      setLatestLog(response.data[0] || null);
    } catch (err) {
      console.error('Error fetching latest log:', err);
      setError('Failed to fetch latest log: ' + (err.response?.data?.detail || err.message));
      if (err.response?.status === 401) {
        console.log('Unauthorized, redirecting to login...');
        navigate('/login');
      }
    }
  };

  const handleVerify = async (e) => {
    e.preventDefault();
    setError('');
    setVerifyResult(null);
    setIsLoading(true);

    const token = localStorage.getItem('token');
    if (!token) {
      setError('Please log in to verify vehicles');
      navigate('/login');
      setIsLoading(false);
      return;
    }

    const plateRegex = /^[A-Z]{2}\d{1,2}[A-Z]{0,2}\d{1,4}$|^[A-Z]{2}\d{3,5}$/;
    if (!plateRegex.test(plateNumber)) {
      setError('Invalid plate format. Use formats like MH12MB8677, GJ01XY1234, or JK12345.');
      setIsLoading(false);
      return;
    }

    try {
      const response = await axios.post(
        `${API_BASE_URL}/vehicles/verify`,
        {
          plate_number: plateNumber,
          confidence: 0.95,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      setVerifyResult(response.data);
      setPlateNumber('');
      fetchLatestLog();
    } catch (err) {
      setError('Failed to verify vehicle: ' + (err.response?.data?.detail || err.message));
      if (err.response?.status === 401) {
        navigate('/login');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleCapture = async (e) => {
    const file = e.target.files[0];
    if (!file) {
      setError('No image selected.');
      return;
    }

    setError('');
    setCaptureResult(null);
    setIsLoading(true);

    const token = localStorage.getItem('token');
    if (!token) {
      setError('Please log in to capture images');
      navigate('/login');
      setIsLoading(false);
      return;
    }

    try {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onloadend = async () => {
        const base64Image = reader.result.split(',')[1];

        try {
          const response = await axios.post(
            `${API_BASE_URL}/vehicles/capture-mobile`,
            {
              image: base64Image,
            },
            {
              headers: {
                Authorization: `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
            }
          );

          setCaptureResult(response.data);
          fetchLatestLog();
        } catch (err) {
          setError('Failed to process image: ' + (err.response?.data?.detail || err.message));
          if (err.response?.status === 401) {
            navigate('/login');
          }
        } finally {
          setIsLoading(false);
        }
      };
      reader.onerror = () => {
        setError('Failed to read the image file.');
        setIsLoading(false);
      };
    } catch (err) {
      setError('Error capturing image: ' + err.message);
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    navigate('/login');
    setIsSidebarOpen(false);
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <PageWrapper>
      <ToggleButton onClick={toggleSidebar}>
        {isSidebarOpen ? <FiX size={24} /> : <FiMenu size={24} />}
      </ToggleButton>
      <Sidebar isOpen={isSidebarOpen}>
        <Logo
          src={sciFiLogo || fallbackLogo}
          alt="Sci-Fi Logo"
          onError={() => console.error('Failed to load logo. Check the path for sciFiLogo or fallbackLogo.')}
        />
        <NavButton onClick={handleLogout}>
          <FiLogOut /> Logout
        </NavButton>
      </Sidebar>
      <Main className="main-content">
        <div className="content-container">
          <h1 className="dashboard-title fade-in">Security Dashboard</h1>

          {error && (
            <p className="error-message fade-in">{error}</p>
          )}

          {/* Latest Access Log Card */}
          <div className="card fade-in">
            <h2 className="text-xl font-semibold mb-4">Latest Access Log</h2>
            {latestLog ? (
              <div className="overflow-x-auto">
                <Table className="access-log-table">
                  <thead>
                    <tr className="table-header">
                      <Th>Plate Number</Th>
                      <Th>Timestamp</Th>
                      <Th>Confidence</Th>
                      <Th>Authorized</Th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="table-row">
                      <Td className="table-cell plate">{latestLog.plate_number}</Td>
                      <Td className="table-cell">{new Date(latestLog.timestamp).toLocaleString()}</Td>
                      <Td className="table-cell">{latestLog.confidence.toFixed(2)}</Td>
                      <Td className="table-cell">
                        <StatusText authorized={latestLog.authorized}>
                          {latestLog.authorized ? '✓' : '✗'}
                        </StatusText>
                      </Td>
                    </tr>
                  </tbody>
                </Table>
              </div>
            ) : (
              <p className="no-logs">No access logs available.</p>
            )}
          </div>

          {/* Capture from Mobile Camera Card */}
          <div className="card fade-in">
            <h2 className="text-xl font-semibold mb-4">Capture from Mobile Camera</h2>
            <div className="capture-container mb-4">
              <FileInput
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleCapture}
                className="input-field"
                disabled={isLoading}
              />
              {isLoading && (
                <div className="spinner-overlay">
                  <Spinner />
                </div>
              )}
            </div>
            {captureResult && (
              <div className="capture-result mt-4">
                <ResultText>
                  Plate: <Highlight>{captureResult.plate_number === 'Not detected' ? 'Not detected' : captureResult.plate_number}</Highlight>
                </ResultText>
                {captureResult.plate_number !== 'Not detected' && (
                  <>
                    <ResultText>
                      Authorized: <StatusText authorized={captureResult.authorized}>{captureResult.authorized ? '✓' : '✗'}</StatusText>
                    </ResultText>
                    <ResultText>
                      Confidence: <Highlight>{captureResult.confidence.toFixed(2)}</Highlight>
                    </ResultText>
                  </>
                )}
                {captureResult.image && (
                  <img
                    src={`data:image/jpeg;base64,${captureResult.image}`}
                    alt="Captured Vehicle"
                    className="capture-image"
                    style={{ width: '100%', borderRadius: '8px', marginTop: '1rem' }}
                  />
                )}
              </div>
            )}
          </div>

          {/* Verify Vehicle Card */}
          <div className="card fade-in">
            <h2 className="text-xl font-semibold mb-4">Verify Vehicle</h2>
            <Form onSubmit={handleVerify} className="verify-form">
              <div className="relative">
                <label htmlFor="plate_number" className="label">
                  Plate Number
                </label>
                <Input
                  id="plate_number"
                  type="text"
                  value={plateNumber}
                  onChange={(e) => setPlateNumber(e.target.value)}
                  className="input-field"
                  placeholder="e.g., MH12MB8677"
                  required
                  disabled={isLoading}
                />
                {isLoading && (
                  <div className="spinner-overlay">
                    <Spinner />
                  </div>
                )}
              </div>
              <Button
                type="submit"
                className="btn-primary"
                disabled={isLoading}
              >
                Verify
              </Button>
            </Form>
            {verifyResult && (
              <div className="verify-result mt-4">
                <ResultText>
                  Plate: <Highlight>{verifyResult.plate_number}</Highlight>
                </ResultText>
                <ResultText>
                  Authorized: <StatusText authorized={verifyResult.authorized}>{verifyResult.authorized ? '✓' : '✗'}</StatusText>
                </ResultText>
              </div>
            )}
          </div>
        </div>
      </Main>
    </PageWrapper>
  );
};

export default SecurityDashboard;