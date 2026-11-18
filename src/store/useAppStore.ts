import { create } from 'zustand';
import type { Horizon, ModelType, PredictionsResponse } from '../api/types';
export type ChartType='line'|'bar'|'stacked'|'area';
interface AppState {
  apiBaseUrl:string; mockMode:boolean; theme:'light'|'dark'|'system';
  selectedApps:string[]; selectedCategories:string[]; selectedMonths:string[];
  excludedMonths:string[]; excludedApps:string[]; chartType:ChartType; view:'actual'|'both'|'forecast'; aggregation:'app'|'total';
  defaultHorizon:Horizon; defaultModel:ModelType|'auto'; lastPrediction:PredictionsResponse|null; recentCommands:string[];
  setApiBaseUrl:(url:string)=>void; setMockMode:(v:boolean)=>void; setTheme:(theme:'light'|'dark'|'system')=>void;
  setFilters:(filters:Partial<Pick<AppState,'selectedApps'|'selectedCategories'|'selectedMonths'|'excludedMonths'|'excludedApps'|'chartType'|'view'|'aggregation'>>)=>void;
  setDefaults:(defaults:Partial<Pick<AppState,'defaultHorizon'|'defaultModel'>>)=>void; setLastPrediction:(p:PredictionsResponse)=>void; addCommand:(command:string)=>void;
  resetFilters:(months:string[])=>void; restoreExcluded:()=>void;
}
export const useAppStore=create<AppState>((set)=>({
  apiBaseUrl:import.meta.env.VITE_API_BASE_URL??'http://localhost:8000',mockMode:false,theme:'system',selectedApps:[],selectedCategories:[],selectedMonths:[],excludedMonths:[],excludedApps:[],chartType:'line',view:'both',aggregation:'app',defaultHorizon:6,defaultModel:'auto',lastPrediction:null,recentCommands:[],
  setApiBaseUrl:(apiBaseUrl)=>set({apiBaseUrl}),setMockMode:(mockMode)=>set({mockMode}),setTheme:(theme)=>set({theme}),
  setFilters:(filters)=>set((state)=>({...state,...filters})),setDefaults:(defaults)=>set((state)=>({...state,...defaults})),setLastPrediction:(lastPrediction)=>set({lastPrediction}),addCommand:(command)=>set((state)=>({recentCommands:[command,...state.recentCommands.filter(v=>v!==command)].slice(0,5)})),resetFilters:(months)=>set({selectedApps:[],selectedCategories:[],selectedMonths:months,excludedMonths:[],excludedApps:[]}),restoreExcluded:()=>set({excludedMonths:[],excludedApps:[]})
}));
