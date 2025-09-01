import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import styled, { keyframes } from 'styled-components';
import { FiLogOut, FiTrash2, FiPlus } from 'react-icons/fi';
import sciFiLogo from '../assets/sci-fi-logo.png';

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
`;

// Sidebar
const Sidebar = styled.aside`
  width: 280px;
  background: rgba(255,255,255,0.05);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255,255,255,0.1);
  padding: 2rem;
  display: flex;
  flex-direction: column;
  color: #e0e0e0;
`;
const Logo = styled.img`
  width: 120px;
  height: 120px;
  margin: 0 auto 2rem;
  animation: ${float} 6s ease-in-out infinite;
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
  transition: background 0.3s;
  &:hover { background: rgba(255,255,255,0.1); }
`;

// Main Content
const Main = styled.main`
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
`;
const Section = styled.section`
  margin-bottom: 2rem;
  background: rgba(255,255,255,0.05);
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  color: #e0e0e0;
`;
const SectionTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #00d9a6;
`;

// Forms & Inputs
const Form = styled.form`
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
`;
const Input = styled.input`
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: rgba(255,255,255,0.1);
  color: #fff;
  font-size: 1rem;
  outline: none;
  &::placeholder { color: rgba(255,255,255,0.7); }
`;
const Button = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #00d9a6, #00c096);
  color: #0d0d1a;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.3s;
  &:hover { background: linear-gradient(135deg, #00c096, #00d9a6); }
`;
const DeleteButton = styled.button`
  background: none;
  border: none;
  color: #ff6b6b;
  cursor: pointer;
  transition: color 0.3s;
  &:hover { color: #ff8787; }
`;
const Table = styled.table`
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  color: #e0e0e0;
`;
const Th = styled.th`
  padding: 0.75rem;
  background: rgba(255,255,255,0.1);
  font-weight: 600;
`;
const Td = styled.td`
  padding: 0.75rem;
  border-top: 1px solid rgba(255,255,255,0.1);
`;
const ErrorText = styled.p`
  color: #ff6b6b;
`;

const AdminDashboard = () => {
  const navigate = useNavigate();
  const [logs, setLogs] = useState([]);
  const [vehicles, setVehicles] = useState([]);
  const [photos, setPhotos] = useState([]);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({
    plate_number: '',
    owner_name: '',
    vehicle_type: '',
  });
  const [formError, setFormError] = useState('');

  const API = 'http://localhost:8000';

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      navigate('/login');
      return;
    }

    const headers = { Authorization: `Bearer ${token}` };

    // Fetch access logs (no authentication required)
    const fetchLogs = async () => {
      try {
        const response = await axios.get(`${API}/logs/?limit=5`);
        setLogs(response.data);
      } catch (err) {
        setError('Failed to fetch access logs: ' + (err.response?.data?.detail || err.message));
      }
    };

    // Fetch authorized vehicles
    const fetchVehicles = async () => {
      try {
        const response = await axios.get(`${API}/vehicles`, { headers });
        setVehicles(response.data);
      } catch (err) {
        setError('Failed to fetch authorized vehicles: ' + (err.response?.data?.detail || err.message));
        if (err.response?.status === 401) {
          navigate('/login');
        }
      }
    };

    // Fetch photos
    const fetchPhotos = async () => {
      try {
        const response = await axios.get(`${API}/photos/`, { headers });
        setPhotos(response.data);
      } catch (err) {
        setError('Failed to fetch photos: ' + (err.response?.data?.detail || err.message));
        if (err.response?.status === 401) {
          navigate('/login');
        }
      }
    };

    fetchLogs();
    fetchVehicles();
    fetchPhotos();
  }, [navigate]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const validatePlate = (plate) => {
    const regex = /^[A-Z]{2}\d{1,2}[A-Z]{0,2}\d{1,4}$|^[A-Z]{2}\d{3,5}$/;
    return regex.test(plate);
  };

  const getErrorMessage = (err) => {
    if (err.response?.data?.detail) {
      const detail = err.response.data.detail;
      if (Array.isArray(detail)) {
        return detail.map((item) => item.msg).join(', ');
      }
      return detail;
    }
    return err.message || 'Unknown error';
  };

  const handleAddVehicle = async (e) => {
    e.preventDefault();
    setFormError('');
    setError('');

    const token = localStorage.getItem('token');
    if (!token) {
      setError('Please log in to add vehicles');
      navigate('/login');
      return;
    }

    if (!validatePlate(formData.plate_number)) {
      setFormError('Invalid plate format. Use formats like MH12MB8677, GJ01XY1234, or JK12345.');
      return;
    }

    try {
      await axios.post(`${API}/vehicles`, formData, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      const response = await axios.get(`${API}/vehicles`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setVehicles(response.data);
      setFormData({ plate_number: '', owner_name: '', vehicle_type: '' });
    } catch (err) {
      setError('Failed to add vehicle: ' + getErrorMessage(err));
      if (err.response?.status === 401) {
        navigate('/login');
      }
    }
  };

  const handleDeleteVehicle = async (plate_number) => {
    const token = localStorage.getItem('token');
    if (!token) {
      setError('Please log in to delete vehicles');
      navigate('/login');
      return;
    }

    try {
      await axios.delete(`${API}/vehicles/${plate_number}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      const response = await axios.get(`${API}/vehicles`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setVehicles(response.data);
    } catch (err) {
      setError('Failed to delete vehicle: ' + getErrorMessage(err));
      if (err.response?.status === 401) {
        navigate('/login');
      }
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    navigate('/login');
  };

  return (
    <PageWrapper>
      <Sidebar>
        <Logo src={sciFiLogo} alt="Sci-Fi Logo" />
        <NavButton onClick={handleLogout}><FiLogOut /> Logout</NavButton>
      </Sidebar>
      <Main>
        <Section>
          <SectionTitle>Welcome, Admin</SectionTitle>
          {error && <ErrorText>{error}</ErrorText>}
          {photos.length === 0 && <ErrorText>Failed to fetch photos: Only managers can view photos</ErrorText>}
        </Section>

        <Section>
          <SectionTitle>Add Authorized Vehicle</SectionTitle>
          {formError && <ErrorText>{formError}</ErrorText>}
          <Form onSubmit={handleAddVehicle}>
            <Input
              name="plate_number"
              placeholder="Plate e.g., MH12MB8677"
              value={formData.plate_number}
              onChange={handleInputChange}
              required
            />
            <Input
              name="owner_name"
              placeholder="Owner Name e.g., John Doe"
              value={formData.owner_name}
              onChange={handleInputChange}
              required
            />
            <Input
              name="vehicle_type"
              placeholder="Type e.g., Car"
              value={formData.vehicle_type}
              onChange={handleInputChange}
              required
            />
            <Button type="submit"><FiPlus /> Add Vehicle</Button>
          </Form>
        </Section>

        <Section>
          <SectionTitle>Recent Access Logs</SectionTitle>
          {logs.length === 0 ? (
            <p>No access logs available.</p>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>Plate Number</Th>
                  <Th>Timestamp</Th>
                  <Th>Confidence</Th>
                  <Th>Authorized</Th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id}>
                    <Td>{log.plate_number}</Td>
                    <Td>{new Date(log.timestamp).toLocaleString()}</Td>
                    <Td>{log.confidence.toFixed(2)}</Td>
                    <Td>{log.authorized ? '✓' : '✗'}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Section>

        <Section>
          <SectionTitle>Authorized Vehicles</SectionTitle>
          {vehicles.length === 0 ? (
            <p>No authorized vehicles available.</p>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>Plate Number</Th>
                  <Th>Owner Name</Th>
                  <Th>Vehicle Type</Th>
                  <Th>Added At</Th>
                  <Th>Actions</Th>
                </tr>
              </thead>
              <tbody>
                {vehicles.map((vehicle) => (
                  <tr key={vehicle.plate_number}>
                    <Td>{vehicle.plate_number}</Td>
                    <Td>{vehicle.owner_name}</Td>
                    <Td>{vehicle.vehicle_type}</Td>
                    <Td>{new Date(vehicle.added_at).toLocaleString()}</Td>
                    <Td>
                      <DeleteButton onClick={() => handleDeleteVehicle(vehicle.plate_number)}>
                        <FiTrash2 />
                      </DeleteButton>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Section>

        <Section>
          <SectionTitle>Photos</SectionTitle>
          {photos.length > 0 ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
              {photos.map((photo) => (
                <div key={photo.id} style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '8px' }}>
                  <img
                    src={`data:image/jpeg;base64,${photo.image_data}`}
                    alt={photo.plate_number}
                    style={{ width: '100%', borderRadius: '4px' }}
                  />
                  <p>{photo.plate_number} at {new Date(photo.timestamp).toLocaleString()}</p>
                </div>
              ))}
            </div>
          ) : (
            <p>No photos available.</p>
          )}
        </Section>
      </Main>
    </PageWrapper>
  );
};

export default AdminDashboard;