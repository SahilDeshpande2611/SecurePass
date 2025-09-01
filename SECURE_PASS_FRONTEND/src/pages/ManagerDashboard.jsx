import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import styled, { keyframes } from 'styled-components';
import { FiLogOut } from 'react-icons/fi';
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
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
const Select = styled.select`
  padding: 0.75rem 1rem;
  border: none;
  border-radius: 8px;
  background: rgba(255,255,255,0.1);
  color: #fff;
  font-size: 1rem;
  outline: none;
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

const ManagerDashboard = () => {
  const navigate = useNavigate();
  const [logs, setLogs] = useState([]);
  const [vehicles, setVehicles] = useState([]);
  const [filteredLogs, setFilteredLogs] = useState([]);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [authorizedFilter, setAuthorizedFilter] = useState('all');
  const [entriesToday, setEntriesToday] = useState(0);
  const [error, setError] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      navigate('/login');
      return;
    }

    // Fetch data initially
    fetchLogs();
    fetchVehicles(token);
  }, [navigate]);

  const fetchLogs = async () => {
    try {
      const response = await axios.get('http://localhost:8000/logs/?limit=10');
      const logsData = response.data;
      setLogs(logsData);
      setFilteredLogs(logsData);

      // Calculate entries for today
      const today = new Date().toISOString().split('T')[0];
      const todayLogs = logsData.filter(log => new Date(log.timestamp).toISOString().split('T')[0] === today);
      setEntriesToday(todayLogs.length);
    } catch (err) {
      setError('Failed to fetch logs: ' + (err.response?.data?.detail || err.message));
    }
  };

  const fetchVehicles = async (token) => {
    try {
      const response = await axios.get('http://localhost:8000/vehicles', {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      setVehicles(response.data);
    } catch (err) {
      setError('Failed to fetch vehicles: ' + (err.response?.data?.detail || err.message));
      if (err.response?.status === 401) {
        navigate('/login');
      }
    }
  };

  const handleFilter = () => {
    let filtered = [...logs];

    // Filter by date range
    if (startDate) {
      filtered = filtered.filter(log => new Date(log.timestamp) >= new Date(startDate));
    }
    if (endDate) {
      filtered = filtered.filter(log => new Date(log.timestamp) <= new Date(endDate));
    }

    // Filter by authorized status
    if (authorizedFilter !== 'all') {
      const isAuthorized = authorizedFilter === 'authorized';
      filtered = filtered.filter(log => log.authorized === isAuthorized);
    }

    setFilteredLogs(filtered);
  };

  const handleLogout = () => {
    localStorage.clear();
    navigate('/login');
  };

  return (
    <PageWrapper>
      <Sidebar>
        <Logo src={sciFiLogo} alt="Sci-Fi Logo" />
        <NavButton onClick={handleLogout}>
          <FiLogOut /> Logout
        </NavButton>
      </Sidebar>
      <Main>
        <Section>
          <SectionTitle>Manager Dashboard</SectionTitle>
          {error && <ErrorText>{error}</ErrorText>}
        </Section>

        <Section>
          <SectionTitle>Today’s Entries</SectionTitle>
          <p>
            Total Entries/Exits Today: <span>{entriesToday}</span>
          </p>
        </Section>

        <Section>
          <SectionTitle>Access Logs</SectionTitle>
          <Form onSubmit={(e) => { e.preventDefault(); handleFilter(); }}>
            <div>
              <label htmlFor="startDate">Start Date</label>
              <Input
                id="startDate"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="endDate">End Date</label>
              <Input
                id="endDate"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="authorizedFilter">Authorized</label>
              <Select
                id="authorizedFilter"
                value={authorizedFilter}
                onChange={(e) => setAuthorizedFilter(e.target.value)}
              >
                <option value="all">All</option>
                <option value="authorized">Authorized</option>
                <option value="unauthorized">Unauthorized</option>
              </Select>
            </div>
            <Button type="submit">Apply Filters</Button>
          </Form>
          {filteredLogs.length > 0 ? (
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
                {filteredLogs.map((log) => (
                  <tr key={log.id}>
                    <Td>{log.plate_number}</Td>
                    <Td>{new Date(log.timestamp).toLocaleString()}</Td>
                    <Td>{log.confidence.toFixed(2)}</Td>
                    <Td>{log.authorized ? '✓' : '✗'}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ) : (
            <p>No access logs available.</p>
          )}
        </Section>

        <Section>
          <SectionTitle>Authorized Vehicles</SectionTitle>
          {vehicles.length > 0 ? (
            <Table>
              <thead>
                <tr>
                  <Th>Plate Number</Th>
                  <Th>Owner Name</Th>
                  <Th>Vehicle Type</Th>
                </tr>
              </thead>
              <tbody>
                {vehicles.map((vehicle) => (
                  <tr key={vehicle.plate_number}>
                    <Td>{vehicle.plate_number}</Td>
                    <Td>{vehicle.owner_name}</Td>
                    <Td>{vehicle.vehicle_type}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ) : (
            <p>No authorized vehicles available.</p>
          )}
        </Section>
      </Main>
    </PageWrapper>
  );
};

export default ManagerDashboard;