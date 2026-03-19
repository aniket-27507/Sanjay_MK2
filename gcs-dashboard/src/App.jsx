import React, { useState } from 'react';
import useWebSocket from './hooks/useWebSocket';
import Sidebar from './components/layout/Sidebar';
import TopBar from './components/layout/TopBar';
import SituationalMap from './components/map/SituationalMap';
import CameraFeedGrid from './components/feeds/CameraFeedGrid';
import StampedeRiskPanel from './components/crowd/StampedeRiskPanel';
import IncidentCommand from './components/command/IncidentCommand';
import ZoneEditor from './components/command/ZoneEditor';
import EvidencePanel from './components/evidence/EvidencePanel';
import AuditLogViewer from './components/audit/AuditLogViewer';
import ScenarioPanel from './components/scenarios/ScenarioPanel';

export default function App() {
  const { connected, latencyMs, send } = useWebSocket();
  const [activeTab, setActiveTab] = useState('map');

  const renderTab = () => {
    switch (activeTab) {
      case 'map':
        return <SituationalMap />;
      case 'cameras':
        return <CameraFeedGrid />;
      case 'crowd':
        return <StampedeRiskPanel />;
      case 'command':
        return <IncidentCommand wsSend={send} />;
      case 'zones':
        return <ZoneEditor wsSend={send} />;
      case 'evidence':
        return <EvidencePanel wsSend={send} />;
      case 'audit':
        return <AuditLogViewer />;
      case 'scenarios':
        return <ScenarioPanel wsSend={send} />;
      default:
        return <SituationalMap />;
    }
  };

  return (
    <div className="app-shell">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="app-main">
        <TopBar connected={connected} latencyMs={latencyMs} />
        <div className="app-content">{renderTab()}</div>
      </div>
    </div>
  );
}
